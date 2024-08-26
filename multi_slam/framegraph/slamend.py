import cuda_ba
import gin
import numpy as np
import torch
from einops import *
from fastcore.all import store_attr

from ..msdp_dpvo import altcorr, fastba
from .patchgraph import PatchGraph
from ..msdp_dpvo.utils import Timer


from ..msdp_dpvo import projective_ops as pops

from multi_slam.msdp_dpvo.lietorch import SE3, SO3, Sim3


IKW = dict(dtype=torch.long, device='cuda')
FKW = dict(dtype=torch.float, device='cuda')
WDEF = np.nan
TDEF = np.nan
FDEF = np.nan

class SubgraphBase:

    def __init__(self, DIM, device, dtype, update_op, graph: PatchGraph):
        store_attr()
        self.graph = graph
        self.update_op = update_op
        self.net = torch.empty(0, self.DIM, device=self.device, dtype=self.dtype)
        self.residuals = torch.empty(0, 2, device=self.device, dtype=self.dtype)

    @gin.configurable('slamend_corr')
    def corr(self, coords, indices, radius, dropout=0.2):
        kk, jj = indices
        coords = rearrange(coords, 'N uv p1 p2 -> 1 N uv p1 p2', uv=2, p1=self.P, p2=self.P)
        corrs = []
        for i in [2, 4, 8, 16, 32, 64]:
            gmap = rearrange(self.graph.gmaps[str(min(i, 8))], 'N M ... -> 1 (N M) ...')
            fmap = self.fmaps[str(i)][None]
            corrs += [ altcorr.corr(gmap, fmap, coords * (8 / i), kk, jj, radius, dropout) ]
        corrs = torch.stack(corrs, dim=2)
        return rearrange(corrs, '1 N lvl g1 g2 p1 p2 -> N (lvl g1 g2 p1 p2)')

    @property
    def P(self):
        return self.graph.P

    def get_opt_edges(self):
        raise NotImplementedError()

    @gin.configurable('fs_update')
    def run_update_op(self, ba_iters=2):
        coords = self.graph.reproject((self.ii, self.jj, self.kk))
        ctx = self.graph.imap.flatten(0, 1)[self.kk]
        with torch.autocast(device_type="cuda"):
            corr = self.corr(coords, (self.kk, self.fjj)).float()
            assert parse_shape(self.net, 'E _') == parse_shape(ctx, 'E _') == parse_shape(corr, 'E _') == parse_shape(self.residuals, 'E _')
            self.net, (delta, _, self.weight) = self.update_op(self.net, ctx, corr, None, self.ii, self.jj, self.kk)


        self.target = coords[...,self.P//2,self.P//2] + delta

        ii, jj, kk, target, weight = self.get_opt_edges()

        poses = self.graph.poses.view(1, -1, 7)
        patches = self.graph.patches.view(1, -1, 3, 3, 3)
        intrinsics = self.graph.intrinsics.view(1, -1, 4)

        lmbda = torch.as_tensor([1e-4], device="cuda")

        t0, t1 = self.get_opt_window()
        is_backend = (self.__class__.__name__ == "Backend")
        fastba.BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M=96, iterations=ba_iters, eff_impl=is_backend)

        # compute colors and points

        # # update points based on new poses and new depth in patches
        points = pops.point_cloud(SE3(self.graph.poses.view(1,-1,7)), self.graph.patches.view(1,-1,3,3,3)[:, :self.graph.m], self.graph.intrinsics.view(1,-1,4), self.graph.ix.view(-1)[:self.graph.m])

        # print("points.shape  : ", points.shape)
        #
        # we divide all coords x y z by the fourth coords to get the 3D coords X Y Z and reshape to keep only X Y Z
        points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)

        # print("points.shape : ", points.shape)
        # print("len(points) : ", len(points))
        # print("self.graph.points_.shape : ", self.graph.points_.shape)

        self.graph.points_[:len(points)] = points[:]

