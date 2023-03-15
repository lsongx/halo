import os
import numpy as np
from numpy.lib.function_base import interp
from scipy.spatial.kdtree import distance_matrix
import torch
from torch.utils.data import Dataset

from mmcv.utils import build_from_cfg
from .builder import DATASETS, LOADERS
from .utils import collect_rays


@DATASETS.register_module()
class MPILLFFDataset(Dataset):
    def __init__(self, size, planes, loader, split, holdout=8, offset=400):
        super().__init__()
        self.size = size
        self.loader = build_from_cfg(loader, LOADERS)
        self.w, self.h = self.loader.w, self.loader.h
        self.split = split
        self.holdout = holdout

        if self.split == 'train':
            rays_ori = [rays_o for i, rays_o in enumerate(self.loader.rays_ori) if i % holdout != 0]
            rays_dir = [rays_d for i, rays_d in enumerate(self.loader.rays_dir) if i % holdout != 0]
            rays_color = [rays_c for i, rays_c in enumerate(self.loader.rays_color) if i % holdout != 0]
            poses = [pose for i, pose in enumerate(self.loader.poses) if i % holdout != 0]
        else:
            rays_ori = [rays_o for i, rays_o in enumerate(self.loader.rays_ori) if i % holdout == 0]
            rays_dir = [rays_d for i, rays_d in enumerate(self.loader.rays_dir) if i % holdout == 0]
            rays_color = [rays_c for i, rays_c in enumerate(self.loader.rays_color) if i % holdout == 0]
            poses = [pose for i, pose in enumerate(self.loader.poses) if i % holdout == 0]
        
        self.rays_ori = np.stack(rays_ori, axis=0).astype('float32')
        self.rays_dir = np.stack(rays_dir, axis=0).astype('float32')
        self.rays_color = np.stack(rays_color, axis=0).astype('float32')
        self.poses = np.stack(poses, axis=0).astype('float32')

        self.dmin = 1.3
        self.dmax = 8.0
        self.ref_id = 12
        self.offset = offset
        self.get_ref_planes(planes)

        if self.size > 0:
            # patch sample
            patch_row_split_num = self.h//self.size
            patch_col_split_num = self.w//self.size
            patch_idx = []
            for i in range(patch_row_split_num):
                for j in range(patch_col_split_num):
                    patch_idx.append([i*self.size, j*self.size])
            if self.h%self.size > 0:
                for j in range(patch_col_split_num):
                    patch_idx.append([self.h-self.size, j])
            if self.w%self.size > 0:
                for i in range(patch_row_split_num):
                    patch_idx.append([i, self.w-self.size])
            if self.h%self.size > 0 and self.w%self.size > 0:
                patch_idx.append([self.h-self.size, self.w-self.size])
            self.patch_idx = patch_idx
        else:
            self.patch_idx = []

        self.compute_all_intersections()

    def get_ref_planes(self, planes):
        h, w, focal = map(int, self.poses[0, :3, -1])
        c2w = self.loader.poses[self.ref_id, :3, :4]
        ref_rays_o, ref_rays_d = collect_rays(h+self.offset, w+self.offset, focal, c2w)
        self.coord = {}
        self.plane_normal = {}
        self.plane_point = {}
        self.proj_to_grid = {}
        for p in range(planes):
            d = self.dmin + (self.dmax-self.dmin)*p/planes
            self.coord[p] = (ref_rays_o + d * ref_rays_d).reshape([-1,3])
            self.plane_point[p] = self.coord[p][0]
            self.plane_normal[p] = np.cross(self.coord[p][0], self.coord[p][1])
            self.proj_to_grid[p] = self.get_grid_proj(self.coord[p], [h+self.offset, w+self.offset])
        return

    def get_grid_proj(self, coord, shape):
        coord = coord.reshape(shape+[3])
        mid_h, mid_w = int(shape[0]//2), int(shape[1]//2)
        mid_point = [coord[mid_h-1, mid_w+1],
                     coord[mid_h-1, mid_w],
                     coord[mid_h, mid_w+1],]
        x_mat = np.stack(mid_point, axis=0).transpose(1,0)
        y_mat = np.array([[1,1,0],[1,0,0],[0,1,0]]).transpose(1,0)
        return y_mat @ np.linalg.inv(x_mat)

    def compute_all_intersections(self):
        self.intersections = []
        self.sample_radius = []
        for rays_o, rays_d in zip(self.rays_ori, self.rays_dir):
            rays_o = rays_o.reshape([-1,3])
            rays_d = rays_d.reshape([-1,3])
            ray_point = rays_o + rays_d
            inter_point = {}
            for p, n in self.plane_normal.items():
                dot = rays_d @ n
                if np.any(np.abs(dot) < 1e-6):
                    raise RuntimeError("no intersection")
                w = ray_point - self.plane_point[p]
                si = -(w@n) / dot
                inter_point[p] = w + si[:,None]*rays_d + self.plane_point[p]
            self.intersections.append(self.get_sample_radius(inter_point))

    def get_sample_radius(self, inter_point):
        # rotate, shift, minus
        for p, ip in inter_point.items():
            proj_coord = (self.proj_to_grid[p]@ip.transpose(1,0)).transpose(1,0)
            proj_coord_hw = proj_coord.reshape([self.h, self.w, 3])
            import pdb;pdb.set_trace()
            theta = -np.arctan2( *((proj_coord_hw[0,self.w-1]-proj_coord_hw[0,0])[:2]) )
            rot = np.asarray([[np.cos(theta),np.sin(theta)], [-np.sin(theta),np.cos(theta)]])
            new_coord = (rot @ proj_coord[:,:2].transpose(1,0)).transpose(1,0)
            new_coord_hw = new_coord.reshape([self.h, self.w, 2])
            left_roll = np.roll(proj_coord, 1, 0); left_roll[0,:,:] = 1e5
            np.abs(proj_coord-left_roll)
        return ip, radius

    def __len__(self):
        return self.rays_ori.shape[0] * len(self.patch_idx)

    def __getitem__(self, index):
        ray_idx = index // len(self.patch_idx)
        if self.size > 0:
            r, c = self.patch_idx[index % len(self.patch_idx)]
            rays_ori = self.rays_ori[ray_idx][r:r+self.size, c:c+self.size, :]
            rays_dir = self.rays_dir[ray_idx][r:r+self.size, c:c+self.size, :]
        else:
            rays_ori = self.rays_ori[ray_idx]
            rays_dir = self.rays_dir[ray_idx]


    ## for rays_ori and rays_dir
    # def __getitem__(self, idx):
    #     ray_idx = idx // len(self.patch_idx)
    #     if self.size > 0:
    #         r, c = self.patch_idx[idx % len(self.patch_idx)]
    #         rays_ori = self.rays_ori[ray_idx][r:r+self.size, c:c+self.size, :]
    #         rays_dir = self.rays_dir[ray_idx][r:r+self.size, c:c+self.size, :]
    #     else:
    #         rays_ori = self.rays_ori[ray_idx]
    #         rays_dir = self.rays_dir[ray_idx]
