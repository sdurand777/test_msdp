
from viewerx import Viewer

from hloc.match_features import WorkQueue
from viewerx import Viewer
import torch
import time

import argparse
from datetime import datetime
from pathlib import Path

import gin
import numpy as np
import torch
from einops import *
from tqdm import tqdm

from dataloaders.ETH3D import ETH3D
from dataloaders.IVM import IVM
from multi_slam.fullsystem import FullSystem
from multi_slam.locnet import LocNet
from multi_slam.MultiTrajectory import MultiTrajectory

from matplotlib import pyplot as plt


GROUPS = {
    'bassin': ['part1', 'part2'],
    # 'table': ['table_3', 'table_4'],
    # 'sofa': ['sofa_1', 'sofa_2', 'sofa_3', 'sofa_4'],
    # 'einstein': ['einstein_1', 'einstein_2'],
    # 'planar': ['planar_2', 'planar_3']
}



# class ViewerX:
#     def __init__(self):
# # data for viewer
#         self.intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")
#         self.image_ = torch.zeros(448, 736, 3, dtype=torch.uint8, device="cpu") 
#         self.poses_ =  torch.zeros(100, 7, dtype=torch.float, device="cuda")
#         # initialize poses to identity matrix
#         self.poses_[:,6] = 1.0
#         self.points_ = torch.zeros(96*100, 3, dtype=torch.float, device="cuda")
#         self.colors_ = torch.zeros(100, 96, 3, dtype=torch.uint8, device="cuda")
#         # define Viewer to render SLAM
#         self.viewer = Viewer(
#             self.image_,
#             self.poses_,
#             self.points_,
#             self.colors_,
#             self.intrinsics_)
#         #time.sleep(1)
#
#     def __call__(self, image):
#
#         if self.viewer is not None:
#             print("viewer update : ")
#             #image = torch.rand(480,640,3)
#             self.viewer.update_image(image)


@torch.no_grad()
def main(group_name):

    torch.manual_seed(1234)
    np.random.seed(1234)

    #gt_mt = MultiTrajectory("Ground_Truth")
    pred_mt = MultiTrajectory("Estimated")


    scenes = [(s, IVM(f"data/IVM/pilier/{s}", i, stride=1, rev=(i%2 == 1))) for i,s in enumerate(GROUPS[group_name])]
    
    #scenes = [(s, IVM(f"data/IVM/bassin/{s}", i, stride=1, rev=(i%2 == 1))) for i,s in enumerate(GROUPS[group_name])]

    print("scenes : \n", scenes)

    # for scene_name, scene_obj in scenes:
    #     for (gt_pose, _, tstamp, _) in scene_obj:
    #         if gt_pose is not None:
    #             gt_mt.insert(scene_name, tstamp, gt_pose)

    print("-- load twoview system")

    twoview_system = LocNet().cuda().eval()
    twoview_system.load_weights("twoview.pth")

    print("-- load odometry system")        

    vo_system = LocNet().cuda().eval()
    vo_system.load_weights("vo.pth")

    print("-- build system")

    model = FullSystem(vo_system, twoview_system)

    # viewer = None
    # if viewer is None:
    #     viewer = ViewerX()


    tstamp = 0


    start_time = datetime.now()
    for scene_name, scene_obj in scenes:

        print("-------------> ADD NEW VIDEO")
        model.add_new_video(scene_name, len(scene_obj), (448,736))

        print("---------------> NEW VIDEO ADDED")

        #for _, intrinsics, tstamp, rgb_path in tqdm(scene_obj):
        for intrinsics, rgb_path in tqdm(scene_obj):

            print("rgb_path : ", rgb_path)

            intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')

            #print("intrinsics : \n", intrinsics)
    
            image = scene_obj.read_image(rgb_path).squeeze(0)

            #print("image.shape : ", image.shape)

            # conversion image opencv from read_image to tensor
            #image = torch.from_numpy(image).permute(2,0,1).cuda()
            #image = image.permute(2,0,1).cuda()
            image = image.to(torch.uint8).cuda()

            # print("image.shape : ", image.shape)
            #
            # viewer(image)

            # image_plt = image.cpu().permute(1,2,0).numpy()
            #
            # image_plt = 2 * image_plt / 255.0
            #
            # # print("image_plt.shape : ", image_plt.shape)
            # # print("np.min(image_plt) : ", np.min(image_plt))
            # # print("np.max(image_plt) : ", np.max(image_plt))
            # # print("image_plt : \n", image_plt)
            #
            # plt.imshow(image_plt)
            # plt.show()
            model.insert_frame(image, intrinsics, tstamp)

            tstamp += 1

        model.complete_video()
        model.backend_update(iters=10)

    while True:
        print("wait ...")
        time.sleep(10)


    results = model.terminate()
    end_time = datetime.now()

    base_dir = Path("our_predictions") / group_name
    base_dir.mkdir(exist_ok=True, parents=True)

    for scene_name, tstamp, pred_pose in results:
        pred_mt.insert(scene_name, tstamp, pred_pose)

    MultiTrajectory.plot_both(pred_mt, gt_mt, save_dir=base_dir)

    rmse_tr_err, rot_err, recalls = MultiTrajectory.error(pred_mt, gt_mt)
    text = f'Err (t): {rmse_tr_err:.03f}m | Err (R): {rot_err:.01f} deg | Recall {recalls} | {end_time-start_time}'
    print(text)
    (base_dir / "results.txt").write_text(text)




# class ViewerX:
#     def __init__(self):
# # data for viewer
#         self.intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")
#         self.image_ = torch.zeros(480, 640, 3, dtype=torch.uint8, device="cpu") 
#         self.poses_ =  torch.zeros(100, 7, dtype=torch.float, device="cuda")
#         # initialize poses to identity matrix
#         self.poses_[:,6] = 1.0
#         self.points_ = torch.zeros(96*100, 3, dtype=torch.float, device="cuda")
#         self.colors_ = torch.zeros(100, 96, 3, dtype=torch.uint8, device="cuda")
#         # define Viewer to render SLAM
#         self.viewer = Viewer(
#             self.image_,
#             self.poses_,
#             self.points_,
#             self.colors_,
#             self.intrinsics_)
#         time.sleep(1)
#
#     def __call__(self):
#
#         if self.viewer is not None:
#             print("viewer update : ")
#             image = torch.rand(480,640,3)
#             self.viewer.update_image(image)
#             # update pose
# # Définir les paramètres de la distribution normale
#             mean = 10.0
#             std = 10.0  # Écart type pour contrôler l'amplitude du bruit
# # Générer le bruit avec la même taille que self.poses_
#             noise = torch.normal(mean=mean, std=std, size=self.poses_.size(), device=self.poses_.device)
# # Ajouter le bruit au tensor de poses
#             self.poses_ = self.poses_ + noise


if __name__ == "__main__":


# # data for viewer
#     intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")
#     image_ = torch.zeros(480, 640, 3, dtype=torch.uint8, device="cpu") 
#     poses_ =  torch.zeros(100, 7, dtype=torch.float, device="cuda")
# # initialize poses to identity matrix
#     poses_[:,6] = 1.0
#     points_ = torch.zeros(96*100, 3, dtype=torch.float, device="cuda")
#     colors_ = torch.zeros(100, 96, 3, dtype=torch.uint8, device="cuda")
#
# # define Viewer to render SLAM
#     viewer = Viewer(
#         image_,
#         poses_,
#         points_,
#         colors_,
#         intrinsics_)
#
#     while True:
#         print("en cours ...")
#         time.sleep(10)

    parser = argparse.ArgumentParser()
    parser.add_argument('group', help='group_name', choices=list(GROUPS.keys()))
    args = parser.parse_args()

    gconfigs = [next(iter(Path('gconfigs').rglob(g)), None) for g in (["model/base.gin", "fullsystem.gin"])]
    assert all(gconfigs)
    gin.parse_config_files_and_bindings(gconfigs, [])

    # twoview_system = LocNet().cuda().eval()
    # twoview_system.load_weights("twoview.pth")
    #
    # vo_system = LocNet().cuda().eval()
    # vo_system.load_weights("vo.pth")
    #
    # model = FullSystem(vo_system, twoview_system)


    # viewer = None
    #
    # while True:
    #     if viewer is None:
    #         viewer = ViewerX()
    #     print("sleep ...")
    #     time.sleep(1)
    #     viewer()


    # while True:
    #     print("sleep ...")
    #     time.sleep(1)


    main(args.group)
