
from numpy import who
from viewerx import Viewer

import gin
import torch
from einops import *
from torch.nn import functional as F

from .msdp_dpvo.lietorch import Sim3
from .locnet import DIM
from .framegraph import Backend, Frontend, PatchGraph, PatchGraphUnion

from matplotlib import pyplot as plt

import numpy as np

import time

@gin.configurable
class FullSystem:

    def __init__(self, locnet, twoview_locnet, M):
        self.P = locnet.P
        self.twoview_network = twoview_locnet
        self.network = locnet
        self.network.eval()
        self.M = M
        self.device = locnet.device
        self.count = 0

        self.all_graphs = []
        self.frontend: Frontend = None
        self.graph: PatchGraph = None
        self.time_since_merge_attempt = 0
        
        # viewer
        self.intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")
        self.image_ = torch.zeros(448, 736, 3, dtype=torch.uint8, device="cpu") 
        self.poses_ =  torch.zeros(40000, 7, dtype=torch.float, device="cuda")
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0
        self.points_ = torch.zeros(M * 40000, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(40000, M, 3, dtype=torch.uint8, device="cuda")

        #self.id_graph_ = torch.tensor(len(self.all_graphs), dtype=torch.int32, device="cuda")

        self.id_graph_ = torch.zeros(2, dtype=torch.int32, device="cuda")

        # define Viewer to render SLAM

        self.viewer = None
        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            self.intrinsics_,
            self.id_graph_)



    def add_new_video(self, name, N, frame_size):
        assert (self.graph is None) or self.graph.is_complete()
        self.graph = PatchGraph(N, self.M, name, frame_size)
        self.all_graphs.append(self.graph)

        print("-----> create frontend")
        self.frontend = Frontend(DIM, self.device, torch.float, self.network.update, self.network.patchify, self.graph)

        print("--------> end of add new video method")


    def complete_video(self):
        assert len(self.all_graphs) == 1
        assert (self.graph is not None) and self.graph.is_complete()
        for itr in range(20):
            self.frontend.run_update_op()
        m = torch.ones_like(self.frontend.ii, dtype=torch.bool)
        self.frontend.remove_factors(m, store=True)
        assert self.frontend.ii.numel() == 0
        del self.frontend
        assert len(self.all_graphs) == 1

    def backend_update(self, iters=20):
        backend = Backend(384, 'cuda', torch.float, self.network.update, self.all_graphs[-1])
        backend.update(iters)
        del backend

    def terminate(self) -> PatchGraph:
        assert len(self.all_graphs) == 1
        graph = self.all_graphs.pop()
        return graph.predictions()

    @gin.configurable
    def rel_pose_batch(self, graph_I: PatchGraph, ii, graph_J: PatchGraph, jj, model_params):
        images = torch.stack((graph_I.images[ii], graph_J.images[jj]), dim=1).to(device=self.device)
        intrinsics = torch.stack((graph_I.intrinsics[ii], graph_J.intrinsics[jj]), dim=1) * 8
        centroids = torch.stack((graph_I.patches[ii, :, :, 1, 1], graph_J.patches[jj, :, :, 1, 1]), dim=1)
        centroids[:,:,:,:2] *= 8
        depth_confidence = torch.stack((graph_I.patch_confidence()[ii], graph_J.patch_confidence()[jj]), dim=1)

        start_time = time.time()
        Sim3_r2l, num_inliers = self.twoview_network(images, intrinsics, centroids, **model_params, depth_conf=depth_confidence)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f" Two view network inference globale Execution time: {execution_time} seconds")

        assert num_inliers.ndim == 1

        pi = graph_I.world_poses[ii].cpu()
        pj = graph_J.world_poses[jj].cpu()
        gt_C = Sim3(pi) * Sim3(Sim3_r2l.cpu()) * Sim3(pj.inv())
        num_inliers = num_inliers.double()
        num_inliers += torch.rand_like(num_inliers).mul(0.001)
        return list(zip(asnumpy(num_inliers).tolist(), gt_C))



    @staticmethod
    def _retrieve_image(i, W, desc_I, desc_J):
        ii = torch.arange(i, i+W, device='cuda')

        sims = einsum(desc_I[ii], desc_J, 'w d, n d -> w n')

        block_sims = F.unfold(sims[None,None], kernel_size=(W, 12))
        block_sims = rearrange(block_sims, '1 (si x) B -> B si x', si=W)
        block_sims_max, block_sims_argmax = block_sims.max(2)
        block_sims_red = reduce(block_sims_max, 'B si -> B', 'min')

        # v max value over all batches
        v, idx = block_sims_red.max(0)

        # ids of block_sims
        jj = block_sims_argmax[idx] + idx

        # scores of (ii, jj) edges
        scores = block_sims_max[idx]
        
        return v, (ii, jj), scores


    def _merge_graphs(self, v):
        graph_I, graph_J = self.all_graphs
        assert graph_I.is_complete()

        # Apply Sim3
        graph_J.sim3_graph(v)

        # Union graphs
        self.frontend.graph = self.graph = PatchGraphUnion(graph_I, graph_J)

        # Update frontend
        idx_map = torch.arange(self.frontend.buf_size, device='cuda')
        idx_map = (idx_map + graph_I.N) % self.frontend.buf_size
        self.frontend.fmaps[idx_map] = self.frontend.fmaps.clone()

        # Perform Global BA
        # backend = Backend(384, 'cuda', torch.float, self.network.update, self.graph)
        # backend.update(10)
        # del backend

        self.all_graphs = [self.graph]
        self.graph.normalize()

    @gin.configurable('fs_insert_frame')
    def insert_frame(self, image, intrinsics, tstamp):

        # assert verifications
        assert not self.network.training
        assert parse_shape(image, 'rgb _ _') == dict(rgb=3)
        assert parse_shape(intrinsics, 'f') == dict(f=4)

        # print("image.shape : ", image.shape)
        #
        # image_tmp = image.cpu().unsqueeze(0)
        #
        # print("image_tmp.shape : ", image_tmp.shape)
        #
        # image_resized = F.interpolate(image_tmp, size=(480, 640), mode='bilinear', align_corners=False)
        #
        # print("image_resized.shape : ", image_resized.shape)
        #
        # image_finale = image_resized.squeeze(0).permute(1,2,0)
        #
        # image_finale = 2 * image_finale / 255.0
        #

        # image_finale = image.cpu().to(torch.uint8)
        #
        # print("image_finale.shape : ", image_finale.shape)
        # print(torch.min(image_finale))
        # print(torch.max(image_finale))
        # print("image_finale \n", image_finale)
        #
        # #image_numpy = np.transpose(image_finale.numpy(),(1,2,0))
        #
        # image_numpy = np.transpose(self.graph.images[0].numpy(),(1,2,0))
        #
        # plt.imshow(image_numpy)
        # plt.show()
        
        # tensor = image.clone().cpu()
        # tensor_resized = tensor.unsqueeze(0).float()  # Shape: (1, 3, 448, 700)
        # #tensor_resized = F.interpolate(tensor_resized, size=(480, 640), mode='bilinear', align_corners=False)
        # tensor_resized = F.interpolate(tensor_resized, size=(528, 960), mode='bilinear', align_corners=False)
        # #tensor_resized = F.interpolate(tensor_resized, size=(240, 320), mode='bilinear', align_corners=False)
        # tensor_resized = tensor_resized.squeeze(0).byte()
        # 
        # print(tensor_resized.shape)
        # print(tensor_resized.dtype)
        # print(torch.min(tensor_resized))
        # print(torch.max(tensor_resized))
        # print(tensor_resized)
        #
        # image_numpy = np.transpose(tensor_resized.numpy(), (1,2,0))
        #
        # plt.imshow(image_numpy)
        # plt.show()
        #
        # h, w, _ = image_numpy.shape
        # image_numpy = image_numpy[:h-h%16, :w-w%16]
        #
        # image_test = torch.from_numpy(image_numpy).permute(2,0,1).cuda()

        # print("image : \n", image)
        # print("image.shape : ", image.shape)
    
        if self.viewer is not None:
            self.viewer.update_image(image)


        start_time = time.time()
        self.frontend(image, tstamp, intrinsics)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f" FRONTEND Execution time: {execution_time} seconds")

        if len(self.all_graphs)==1:
            id_tmp = torch.tensor([self.all_graphs[0].n , 0], dtype=torch.int32, device="cuda")
            self.id_graph_.copy_(id_tmp)
        elif len(self.all_graphs)==2:
            id_tmp = torch.tensor([self.all_graphs[0].n , self.frontend.graph.poses.shape[0]], dtype=torch.int32, device="cuda")
            self.id_graph_.copy_(id_tmp)
        
        #self.id_graph_.copy_(torch.tensor(len(self.all_graphs)))
        
        # on a deux graphs en cours nouvelle session
        if len(self.all_graphs) == 2:
            num_poses_graph0 = self.all_graphs[0].n

            # print("num_poses_graph0 : ", num_poses_graph0)
            #
            num_poses = self.frontend.graph.poses.shape[0] 
            self.poses_[num_poses_graph0:num_poses_graph0+num_poses].copy_(self.frontend.graph.poses)

            # print("num_poses : ", num_poses)
            # print("num_poses_graph0*96 : ",num_poses_graph0*96)
            # print("(num_poses_graph0+num_poses)*96 : ",(num_poses_graph0+num_poses)*96)
            # print("self.frontend.graph.points_.shape : ", self.frontend.graph.points_.shape)

            self.points_[num_poses_graph0*96:num_poses_graph0*96 + num_poses*96].copy_(self.frontend.graph.points_)
            
            num_colors = self.frontend.graph.colors_.shape[0]
            self.colors_[num_colors:num_colors*2].copy_(self.frontend.graph.colors_)
        else:
            num_poses = self.frontend.graph.poses.shape[0] 
            num_points = self.frontend.graph.points_.shape[0]
            num_colors = self.frontend.graph.colors_.shape[0]
            self.poses_[:num_poses].copy_(self.frontend.graph.poses)
            self.points_[:num_points].copy_(self.frontend.graph.points_)
            self.colors_[:num_colors].copy_(self.frontend.graph.colors_)

        # if self.count > 8:
        #     # update poses after tracking    
        #     self.poses_ = self.frontend.graph.poses
        #     print("self.poses_[:10] : ", self.poses_[:10])
        #     time.sleep(10)


        self.time_since_merge_attempt += 1
        self.count += 1

        if (self.graph.ii_inac.numel() > 96*10) and (self.count % 50 == 0):
            # Perform Global BA
            start_time = time.time()
            backend = Backend(384, 'cuda', torch.float, self.network.update, self.graph)
            backend.update(2)
            del backend
            end_time = time.time()
            execution_time = end_time - start_time
            print(f" BACKEND GLOBAL Execution time: {execution_time} seconds")


        assert len(self.all_graphs) in [1,2]
        if (len(self.all_graphs) == 2) and (self.time_since_merge_attempt > 10):

            print("--- START MERGE ATTEMPT ---")

            graph_J, graph_I = self.all_graphs

            RAD = 25 # Only add connections outside the optimization window.
            i = self.graph.n - RAD

            # pas assez de frame dans le nouveau graph
            if i < 0:
                return

            # recuperation des descripteurs globaux pour reconnections
            descsI, descsJ = graph_I.global_desc, graph_J.global_desc[:graph_J.N]

            start_time = time.time()
            retr = self._retrieve_image(i, 6, descsI, descsJ)
            end_time = time.time()
            execution_time = end_time - start_time
            print("retr[0] : ", retr[0])
            print(f" RETRIEVE IMAGE Execution time: {execution_time} seconds")

            if retr[0] > 0.4:
                # get graph associated to max sim value
                _, (ii, jj), scores = retr
            else:
                return

            self.time_since_merge_attempt = 0

            # graph_I.sim3_graph(Sim3.Random(1)[0])
            # graph_J.sim3_graph(Sim3.Random(1)[0])

            start_time = time.time()
            rel_poses = self.rel_pose_batch(graph_J, jj.cpu(), graph_I, ii.cpu())
            end_time = time.time()
            execution_time = end_time - start_time
            print(f" REL POSE BATCH Execution time: {execution_time} seconds")

            inl, v = max(rel_poses)
            if (inl >= 10):
                start_time = time.time()
                self._merge_graphs(v)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f" MERGE GRAPH Execution time: {execution_time} seconds")
                start_time = time.time()
                self.backend_update(iters=5)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"  BACKEND MERGE GRAPH Execution time: {execution_time} seconds")


