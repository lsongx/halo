import random
import numpy as np
import torch
from torch.distributed.distributed_c10d import init_process_group

from .synthetic_dataset import get_rays
from .builder import DATASETS
from .llff_dataset import LLFFDataset
from .aug_llff_dataset import get_normal_proj 


@DATASETS.register_module()
class AugTwoLLFFDataset(LLFFDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uv_plane, st_plane = self.find_uv_st_plane()
        self.uv_plane = uv_plane
        self.st_plane = st_plane

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        rand_pose_a = int(random.random()*self.poses.shape[0])
        rand_pose_b = int(random.random()*self.poses.shape[0])
        weight = random.random()
        aug_pose = self.poses[rand_pose_a,:3,:4]*weight+self.poses[rand_pose_b,:3,:4]*(1-weight)
        aug_rays_ori, aug_rays_dir = get_rays(self.h, self.w, self.focal, aug_pose)
        sample_rays_ori, sample_rays_dir = get_rays(
            self.h, self.w, self.focal, self.poses[rand_pose_b,:3,:4])

        if self.batch_size > 0:
            n = self.h*self.w
            perm = torch.randperm(n)
            select_mask = torch.linspace(0,n-1,n)[perm].long()[:self.batch_size]
            batch['aug_rays_ori'] = aug_rays_ori.reshape([-1,3])[select_mask]
            batch['aug_rays_dir'] = aug_rays_dir.reshape([-1,3])[select_mask]
            batch['sample_rays_ori'] = sample_rays_ori.reshape([-1,3])[select_mask]
            batch['sample_rays_dir'] = sample_rays_dir.reshape([-1,3])[select_mask]
        else:
            batch['aug_rays_ori'] = aug_rays_ori.reshape([-1,3])
            batch['aug_rays_dir'] = aug_rays_dir.reshape([-1,3])
            batch['sample_rays_ori'] = sample_rays_ori.reshape([-1,3])
            batch['sample_rays_dir'] = sample_rays_dir.reshape([-1,3])

        return batch

    def find_uv_st_plane(self):
        pose = torch.tensor(self.all_poses[self.center_pose_idx, :3, :4])
        rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose)
        rays_ori_selected = np.stack([rays_ori[0,0]]*3)
        mid_h = self.h//2
        mid_w = self.w//2
        rays_dir_selected = np.stack([rays_dir[mid_h,mid_w], rays_dir[mid_h,mid_w+1], rays_dir[mid_h+1,mid_w]])

        plane_points = rays_ori_selected + self.near*rays_dir_selected
        uv_plane = get_normal_proj(plane_points)

        plane_points = rays_ori_selected + self.far*rays_dir_selected
        st_plane = get_normal_proj(plane_points)
        return uv_plane, st_plane

