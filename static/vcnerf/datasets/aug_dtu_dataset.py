import os
import math
import numpy as np
from PIL import Image
import torch
import random

from vcnerf.utils import get_root_logger
from .builder import DATASETS
from .dtu_dataset import DTUDataset, gen_rays
from .aug_llff_dataset import get_normal_proj, get_line_plane_collision


@DATASETS.register_module()
class AugDTUDataset(DTUDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uv_plane, st_plane = self.find_uv_st_plane()
        self.uv_plane = uv_plane
        self.st_plane = st_plane

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        # pose = self.poses[idx, :3, :4]
        # aug_axis_degree = (random.random()-0.5)*30+3
        # aug_mat = spatial.transform.Rotation.from_euler('z', aug_axis_degree, degrees=True)
        # aug_mat = torch.tensor(aug_mat.as_matrix(), device=pose.device).float()
        # aug_pose = aug_mat @ pose
        # aug_rays_ori, aug_rays_dir = get_rays(self.h, self.w, self.focal, aug_pose)
        # aug_rays_ori = sample_nearby_point_on_sphere(aug_rays_ori, np.pi/4)
        rand_pose_a = int(random.random()*self.all_poses.shape[0])
        rand_pose_b = int(random.random()*self.all_poses.shape[0])
        weight = random.random()
        aug_pose = self.all_poses[rand_pose_a]*weight+self.all_poses[rand_pose_b]*(1-weight)
        aug_rays_ori, aug_rays_dir = gen_rays(aug_pose[None], self.w, self.h, self.focal, self.c) 

        if self.batch_size > 0:
            n = self.h*self.w
            perm = torch.randperm(n)
            select_mask = torch.linspace(0,n-1,n)[perm].long()[:self.batch_size]
            batch['aug_rays_ori'] = aug_rays_ori.reshape([-1,3])[select_mask]
            batch['aug_rays_dir'] = aug_rays_dir.reshape([-1,3])[select_mask]
        else:
            batch['aug_rays_ori'] = aug_rays_ori.reshape([-1,3])
            batch['aug_rays_dir'] = aug_rays_dir.reshape([-1,3])

        return batch

    def find_uv_st_plane(self):
        pose = torch.tensor(self.poses[0])
        rays_ori, rays_dir = gen_rays(pose[None], self.w, self.h, self.focal, self.c) 
        rays_ori, rays_dir = rays_ori[0], rays_dir[0]
        rays_ori_selected = np.stack([rays_ori[0,0]]*3)
        mid_h = self.h//2
        mid_w = self.w//2

        # rays_dir_selected = np.stack([rays_dir[0,0], rays_dir[self.h-1,0], rays_dir[0,self.w-1]])
        # plane_points = rays_ori_selected + 0.01*rays_dir_selected
        rays_dir_selected = np.stack([rays_dir[mid_h,mid_w], rays_dir[mid_h,mid_w+1], rays_dir[mid_h+1,mid_w]])
        plane_points = rays_ori_selected + self.near*rays_dir_selected
        uv_plane = get_normal_proj(plane_points)

        rays_dir_selected = np.stack([rays_dir[mid_h,mid_w], rays_dir[mid_h,mid_w+1], rays_dir[mid_h+1,mid_w]])
        plane_points = rays_ori_selected + self.far*rays_dir_selected
        st_plane = get_normal_proj(plane_points)

        # use center pose as uv plane
        uv_plane = {'normal': st_plane['normal'], 'point': rays_ori[0,0], 'proj': st_plane['proj']}
        self.center_uv = get_line_plane_collision(rays_ori[0,0,:], rays_dir[0,0,:], uv_plane).reshape([2])

        return uv_plane, st_plane

