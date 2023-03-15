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
        # import pdb;pdb.set_trace()
        # compute all grid can be slow and memory comsuming
        # lets do it on gpu!

    def get_ref_planes(self, planes):
        h, w, focal = map(int, self.loader.poses[0, :3, -1])
        c2w = self.loader.poses[self.ref_id, :3, :4]
        ref_rays_o, ref_rays_d = collect_rays(h+self.offset, w+self.offset, focal, c2w)
        self.grid_bound = {}
        self.plane_normal = {}
        self.plane_point = {}
        self.proj_to_grid = {}
        for p in range(planes):
            d = self.dmin + (self.dmax-self.dmin)*p/planes
            coord = (ref_rays_o + d * ref_rays_d).reshape([-1,3])
            self.plane_point[p] = coord[0]
            self.plane_normal[p] = np.cross(coord[0], coord[1])
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
        return proj

    def get_grid_bound(self, projected_coord):
        h_max, w_max = np.ceil(projected_coord).max(axis=0)[:2]
        h_min, w_min = np.floor(projected_coord).min(axis=0)[:2]
        return h_min, h_max, w_min, w_max

    def compute_all_intersections(self):
        self.intersections = []
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
            self.intersections.append(self.get_grid_weight(inter_point))

    def get_grid_weight(self, inter_point):
        all_weight, all_grid = [], []
        for p, ip in inter_point.items():
            projected_ip = (self.proj_to_grid[p]@ip.T).T
            h_min, h_max, w_min, w_max = self.get_grid_bound(projected_ip)
            bh_i, bh_a, bw_i, bw_a = self.grid_bound[p]
            assert (h_min>bh_i)&(w_min>bw_i)&(w_max<bw_a)&(h_max<bh_a)

            h_ceil, w_ceil = np.ceil(projected_ip)[:,0], np.ceil(projected_ip)[:,1]
            h_floor, w_floor = np.floor(projected_ip)[:,0], np.floor(projected_ip)[:,1]
            vec_p = np.zeros_like(h_floor)+p+1
            weight = []
            weight.append( np.stack(
                [h_ceil, w_ceil, vec_p, (projected_ip[:,0]-h_floor)*(projected_ip[:,1]-w_floor)],1) )
            weight.append( np.stack(
                [h_ceil, w_floor, vec_p, (projected_ip[:,0]-h_floor)*(w_ceil-projected_ip[:,1])],1) )
            weight.append( np.stack(
                [h_floor, w_ceil, vec_p, (h_ceil-projected_ip[:,0])*(projected_ip[:,1]-w_floor)],1) )
            weight.append( np.stack(
                [h_floor, w_floor, vec_p, (h_ceil-projected_ip[:,0])*(w_ceil-projected_ip[:,1])],1) )
            weight = np.stack(weight, 0).transpose([1,0,2])
            # weight: [n_points, 4(corner), 4(coord_idx+weight)]

            all_weight.append(weight[:,:,3:])
            all_grid.append(weight[:,:,:3])
        # all_weight: [n_point, n_plane, 4(n_corner), 1]
        all_weight = np.stack(all_weight, axis=1)
        # all_grid: [n_point, n_plane, 4(n_corner), 3(coord_idx)]
        all_grid = np.stack(all_grid, axis=1)
        return {'weight': all_weight, 'grid': all_grid}


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
