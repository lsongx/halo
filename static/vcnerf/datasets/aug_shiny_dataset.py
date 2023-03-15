import os
import numpy as np
from skimage import io
import torch
import random

from vcnerf.utils import get_root_logger
from .builder import DATASETS
from .aug_llff_dataset import rotation_matrix_from_vectors, get_line_plane_collision
from .shiny_dataset import ShinyDataset, get_rays


def get_normal_proj(plane_points):
    a, b, c = plane_points
    normal = np.cross(a-c, b-c)
    normal = normal/np.linalg.norm(normal)
    # # to a new grid
    # grid_length = np.linalg.norm(a-c)
    # y_mat = np.array([[0,0,0],[0,grid_length,0],[grid_length,0,0]]).T+1
    # proj = y_mat @ np.linalg.inv(plane_points.T)
    # return {'normal': normal, 'point': a, 'proj': proj}
    proj = rotation_matrix_from_vectors(normal, np.array([0,0,1]))
    return {'normal': torch.tensor(normal).float(), 
            'point': torch.tensor(a).float(), 
            'proj': torch.tensor(proj).float()}


@DATASETS.register_module()
class AugShinyDataset(ShinyDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center_pose = self.orbiter_dataset.sfm.ref_img['ori_pose']
        self.ref_fx = self.orbiter_dataset.sfm.ref_cam['fx']
        self.ref_fy = self.orbiter_dataset.sfm.ref_cam['fy']
        self.ref_px = self.orbiter_dataset.sfm.ref_cam['px']
        self.ref_py = self.orbiter_dataset.sfm.ref_cam['py']
        
        self.poses = torch.stack([torch.tensor(i['ori_pose']) for i in self.orbiter_dataset.imgs], 0)
        cams = [i['camera_id'] for i in self.orbiter_dataset.imgs]
        self.fx = [self.orbiter_dataset.sfm.cams[i]['fx'] for i in cams]
        self.fy = [self.orbiter_dataset.sfm.cams[i]['fy'] for i in cams]
        self.px = [self.orbiter_dataset.sfm.cams[i]['px'] for i in cams]
        self.py = [self.orbiter_dataset.sfm.cams[i]['py'] for i in cams]

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
        rand_pose_a = int(random.random()*self.poses.shape[0])
        rand_pose_b = int(random.random()*self.poses.shape[0])
        weight = random.random()
        aug_pose = self.poses[rand_pose_a,:3,:4]*weight+self.poses[rand_pose_b,:3,:4]*(1-weight)
        aug_rays_ori, aug_rays_dir = get_rays(self.h, self.w, 
                                              self.px[0], self.py[0], 
                                              self.fx[0], self.fy[0], 
                                              aug_pose)

        if self.batch_size > 0:
            select_idx = torch.randperm(self.h*self.w)[:self.batch_size]
            batch['aug_rays_ori'] = aug_rays_ori.reshape([-1,3])[select_idx]
            batch['aug_rays_dir'] = aug_rays_dir.reshape([-1,3])[select_idx]
        else:
            batch['aug_rays_ori'] = aug_rays_ori.reshape([-1,3])
            batch['aug_rays_dir'] = aug_rays_dir.reshape([-1,3])

        return batch

    def find_uv_st_plane(self):
        pose = torch.tensor(self.center_pose)
        rays_ori, rays_dir = get_rays(self.h, self.w, 
                                      self.ref_px, self.ref_py,
                                      self.ref_fx, self.ref_fy,
                                      pose)
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
        
        uv_plane = {'normal': st_plane['normal'], 'point': rays_ori[0,0], 'proj': st_plane['proj']}
        self.center_uv = get_line_plane_collision(rays_ori[0,0,:], rays_dir[0,0,:], uv_plane).reshape([2])
        # import pdb;pdb.set_trace()
        # pose = self.poses[0, :3, :4]
        # pose = self.poses[1, :3, :4]
        # rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose)
        # t=torch.tensor(uv_plane['proj']).float() @ (self.near*rays_dir+rays_ori).reshape([-1,3]).T
        # t[:,100:120].T
        return uv_plane, st_plane

