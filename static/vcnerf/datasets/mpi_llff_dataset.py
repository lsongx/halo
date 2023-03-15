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
    def __init__(self, size, planes, loader, split, holdout=8, offset=1000):
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

        self.d_min = 1.3
        self.d_max = 8.0
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
            self.patch_idx = [None]

    def get_ref_planes(self, planes):
        h, w, focal = map(int, self.loader.poses[0, :3, -1])
        c2w = self.loader.poses[self.ref_id, :3, :4]
        ref_rays_o, ref_rays_d = collect_rays(h+self.offset, w+self.offset, focal, c2w)
        self.grid_bound = {}
        self.plane_normal = {}
        self.plane_point = {}
        self.proj_to_grid = {}
        for p in range(planes):
            d = self.d_min + (self.d_max-self.d_min)*p/planes
            coord = (ref_rays_o + d * ref_rays_d).reshape([-1,3]).astype('float32')
            self.plane_point[p] = coord[0]
            vec0 = coord[w]-coord[0]; vec1 = coord[(w-1)*h]-coord[0]
            self.plane_normal[p] = np.cross(vec0, vec1)
            self.proj_to_grid[p] = self.get_grid_proj(coord, [h+self.offset, w+self.offset])
            projected_coord = (self.proj_to_grid[p]@coord.T).T
            self.grid_bound[p] = self.get_grid_bound(projected_coord) # used for sanity check
        return

    def get_grid_proj(self, coord, shape):
        coord = coord.reshape(shape+[3])
        mid_h, mid_w = int(shape[0]//2), int(shape[1]//2)
        mid_point = [coord[mid_h-1, mid_w+1],
                     coord[mid_h-1, mid_w],
                     coord[mid_h, mid_w+1],]
        x_mat = np.stack(mid_point, axis=0).T
        y_mat = np.array([[1,1,0],[1,0,0],[0,1,0]]).T
        proj = y_mat @ np.linalg.inv(x_mat)
        return proj.astype('float32')

    def get_grid_bound(self, projected_coord):
        h_max, w_max = np.ceil(projected_coord).max(axis=0)[:2]
        h_min, w_min = np.floor(projected_coord).min(axis=0)[:2]
        return h_min, h_max, w_min, w_max

    def __len__(self):
        return self.rays_ori.shape[0] * len(self.patch_idx)

    def __getitem__(self, index):
        ray_idx = index // len(self.patch_idx)
        if self.size > 0:
            r, c = self.patch_idx[index % len(self.patch_idx)]
            rays_ori = self.rays_ori[ray_idx][r:r+self.size, c:c+self.size, :]
            rays_dir = self.rays_dir[ray_idx][r:r+self.size, c:c+self.size, :]
            rays_color = self.rays_color[ray_idx][r:r+self.size, c:c+self.size, :]
        else:
            rays_ori = self.rays_ori[ray_idx]
            rays_dir = self.rays_dir[ray_idx]
            rays_color = self.rays_color[ray_idx]

        d = {'rays_ori': rays_ori, 
             'rays_dir': rays_dir, 
             'rays_color': rays_color,     
             'plane_point': self.plane_point, 
             'plane_normal': self.plane_normal, 
             'proj_to_grid': self.proj_to_grid, 
             'size': self.size}
        return d

