from pathlib import Path

import gin
import numpy as np
import torch

from torch.nn import functional as F

from evo.core.sync import matching_time_indices
from evo.tools.file_interface import read_tum_trajectory_file
from imageio.v3 import imread

import cv2

import os

import cv2
import numpy as np
from skimage import color, exposure
import matplotlib.pyplot as plt


KWARGS = dict(dtype=torch.float, device='cuda')

@gin.configurable
class IVM:

    def __init__(self, base_path, indice_scene, stride, rev=False):
        self.base_path = Path(base_path)
        assert self.base_path.exists(), self.base_path
        #traj = read_tum_trajectory_file(self.base_path / "groundtruth.txt")
        #self.intrinsics = np.asarray(list(map(float, (self.base_path / "calibration.txt").read_text().split())))
        # tstamps = []
        # rgb_images = []
        #
        # all_lines = (self.base_path / "rgb.txt").read_text().splitlines()
        # if rev:
        #     all_lines = all_lines[::-1]
        #     print(f"Reversing", self.base_path.name)
        # for idx, line in enumerate(all_lines):
        #     if (idx % stride) != 0:
        #         continue
        #     tstamp, rgb = line.split()
        #     tstamps.append(float(tstamp))
        #     rgb_images.append(self.base_path / rgb)
        #
        # poses = np.array(traj.poses_se3)
        # assignment = dict(zip(*matching_time_indices(tstamps, traj.timestamps)))
        # self.files = []
        # for idx, data in enumerate(zip(tstamps, rgb_images)):
        #     if j := assignment.get(idx, None):
        #         self.files.append((*data, poses[j]))
        #     else:
        #         self.files.append((*data, None))
        print("self.base_path : ", self.base_path)

        photo_files = [file for file in os.listdir(self.base_path) if file.endswith("LRR.JPG")]

        if indice_scene%2 == 0:
            photo_files.sort()
        else:
            photo_files.sort(reverse=True)

        print("photo_files : \n", photo_files)
        self.files = [os.path.join(self.base_path, file) for file in photo_files]

        intrinsics_vec = [4.5990068054199219e+02, 4.5990068054199219e+02, 3.3639562416076660e+02, 2.6883334159851074e+02]
        ht0, wd0 = [376, 514]

        image_size = [448, 736]

        intrinsics = torch.as_tensor(intrinsics_vec)
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        self.intrinsics = intrinsics





    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rgb_path  = self.files[idx]
        return np.copy(self.intrinsics), rgb_path

    #@staticmethod
    def read_image(self, rgb_path):
        img = cv2.imread(rgb_path)

#         # color traitement
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir BGR en RGB
#
# # Convertir l'image en espace de couleur LAB
#         lab_img = color.rgb2lab(img)
#          
# # Séparer les composantes L, a et b
#         L = lab_img[:, :, 0]
#         a = lab_img[:, :, 1]
#         b = lab_img[:, :, 2]
#          
# # Corriger la composante L pour étendre la dynamique
#         L_min = np.min(L)
#         L_max = np.max(L)
#         L = (L - L_min) / (L_max - L_min) * 100  # Remap L to [0, 100]
#          
#         # Recombiner les composantes corrigées
#         corrected_lab_img = np.stack([L, a, b], axis=-1)
#          
#         # Convertir l'image corrigée de LAB à RGB
#         corrected_rgb_img = color.lab2rgb(corrected_lab_img)
#          
#         # Afficher l'image corrigée
#         plt.figure()
#         plt.imshow(corrected_rgb_img)
#         plt.show()
#  
#         img = corrected_rgb_img

        H, W, _ = img.shape
        
        # rectification
        K_r = np.array([4.6011859130859375e+02, 0., 3.0329241943359375e+02, 0., 4.6011859130859375e+02, 2.4381889343261719e+02, 0., 0., 1. ]).reshape(3,3)
        d_r = np.array([-3.34759861e-01, 1.55759037e-01, 7.29110325e-04,
                        1.10754154e-04, -4.32639048e-02 
                        ]).reshape(5)
        R_r = np.array([9.9999679129872809e-01, -2.5332565953597990e-03,
                        1.8083868698132711e-06, 2.5332551721412530e-03,
                        9.9999663218813351e-01, 5.6411933458219384e-04,
                        -3.2374398044074352e-06, -5.6411294338637344e-04,
                        9.9999984088304039e-01 
                        ]).reshape(3,3)

        P_r = np.array([ 4.5990068054199219e+02, 0., 3.3639562416076660e+02,
                        5.9697221843133875e+01, 0., 4.5990068054199219e+02,
                        2.6883334159851074e+02, 0., 0., 0., 1., 0. 
                        ]).reshape(3,4)

        map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (616, 514), cv2.CV_32F)

        intrinsics_vec = [4.5990068054199219e+02, 4.5990068054199219e+02, 3.3639562416076660e+02, 2.6883334159851074e+02]
        ht0, wd0 = [376, 514]

        images = [cv2.remap(img, map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]

        images = torch.from_numpy(np.stack(images, 0))

        images = images.permute(0, 3, 1, 2).to("cuda", dtype=torch.float32)

        # Ensure either size or scale_factor is defined

        image_size = [448, 736]
        #image_size = [514, 616]

        #print("++++++++ image_size  : ",image_size)
        if image_size is not None:
            images = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)
        else:
            raise ValueError("image_size must be defined")
            
        intrinsics = torch.as_tensor(intrinsics_vec)
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        #self.intrinsics = intrinsics

        # #img = img[:(H - H%32), :(W - W%32)]
        #
        # # divisible par 16
        # img = img[:(H - H%16), :(W - W%16)]
        #
        # #image = torch.as_tensor(np.copy(img), **KWARGS).permute(2,0,1)
        # #return image


        return images
