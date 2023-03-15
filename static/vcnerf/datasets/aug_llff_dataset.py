import os
import os.path as osp
import random
import json
from scipy import spatial
import numpy as np
import imageio
import cv2
import torch
from torch.distributed.distributed_c10d import init_process_group

from vcnerf.utils import get_root_logger
from .synthetic_dataset import get_rays
from .builder import DATASETS
from .llff_dataset import LLFFDataset


def sample_nearby_point_on_sphere(points, angle=np.pi/9):
    rad = torch.sqrt(points[...,0]**2+points[...,1]**2)
    full_rad = points.norm(dim=-1)
    theta = torch.atan2(rad, points[...,2])
    theta += (random.random()-0.5)*angle
    rad = full_rad * theta.sin()
    phi = torch.atan2(points[...,1], points[...,0])
    phi += (random.random()-0.5)*angle
    new_points = torch.stack([rad*phi.cos(), rad*phi.sin(), full_rad*theta.cos(),], dim=2)
    return new_points


def rotation_matrix_from_vectors(vec1, vec2):
    # https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


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
    return {'normal': normal, 'point': a, 'proj': proj}


def get_line_plane_collision(rays_ori, rays_dir, plane, epsilon=1e-6):
    # https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
    for k,v in plane.items():
        plane[k] = torch.tensor(v).float().to(rays_ori.device)
    plane_normal, plane_point = plane['normal'], plane['point']
    n_dot_u = (plane_normal[None]*rays_dir).sum(-1)
    if not (n_dot_u.abs()>epsilon).all():
        raise RuntimeError("no intersection or line is within plane")
    w = rays_ori - plane_point
    si = -(plane_normal*w).sum(-1) / n_dot_u
    coord_3d = w + si[:,None] * rays_dir + plane_point[None]
    coord = torch.matmul(plane['proj'], coord_3d.T).T[:,:2]
    # t=coord_3d-coord_3d[:1,:]
    # (t*plane_normal[None]).sum(-1)
    # return coord_3d[...,:2]
    return coord


@DATASETS.register_module()
class AugLLFFDataset(LLFFDataset):
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
        rand_pose_a = int(random.random()*self.poses.shape[0])
        rand_pose_b = int(random.random()*self.poses.shape[0])
        weight = random.random()
        aug_pose = self.poses[rand_pose_a,:3,:4]*weight+self.poses[rand_pose_b,:3,:4]*(1-weight)
        aug_rays_ori, aug_rays_dir = get_rays(self.h, self.w, self.focal, aug_pose)

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
        pose = torch.tensor(self.all_poses[self.center_pose_idx, :3, :4])
        rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose)
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

