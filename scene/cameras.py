#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import math
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix
from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from scene.colmap_loader import qvec2rotmat, rotmat2qvec
from typing import NamedTuple

def fov2focal(fov, pixels):    
    return pixels / (2 * math.tan(fov / 2))

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_id = int(image_name)
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        
        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        # W 2 C
        self.w2c = torch.tensor(getWorld2View2(R, T, trans, scale)).cuda()
        self.world_view_transform = self.w2c.transpose(0, 1)
        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        self.focal_y = self.image_height / (2.0 * tan_fovy)
        self.focal_x = self.image_width / (2.0 * tan_fovx)
         
        self.K = torch.tensor([
            [fov2focal(self.FoVx, self.image_width), 0, 0.5*self.image_width],
            [0, fov2focal(self.FoVy, self.image_height), 0.5*self.image_height],
            [0, 0, 1]]).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        full_proj_transform_cpu = (self.world_view_transform.cpu().unsqueeze(0).bmm(self.projection_matrix.cpu().unsqueeze(0))).squeeze(0)
        self.full_proj_transform = full_proj_transform_cpu.cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_k(self, res=1):
        K = torch.tensor([
        [fov2focal(self.FoVx, self.image_width/res), 0, 0.5*self.image_width/res],
        [0, fov2focal(self.FoVy, self.image_height/res), 0.5*self.image_height/res],
        [0, 0, 1]]).cuda()
        return K
        

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

