from sys import path_importer_cache
import numpy as np
from numpy.lib.utils import _split_line
import torch
from torch.utils.data import Dataset

from mmcv.utils import build_from_cfg
from .utils import convert_rays_to_ndc_rays
from .builder import DATASETS, LOADERS


@DATASETS.register_module()
class NeRFDataset(Dataset):
    def __init__(self, loader, split, holdout=8):
        super().__init__()
        assert split in ('train', 'val', 'test'), split
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

        if self.split == 'train':
            self.rays_ori = np.reshape(self.rays_ori, [-1, 3])
            self.rays_dir = np.reshape(self.rays_dir, [-1, 3])
            self.rays_color = np.reshape(self.rays_color, [-1, 3])

        if self.loader.ndc:
            self.ndc_rays_ori, self.ndc_rays_dir = convert_rays_to_ndc_rays(
                h=self.loader.h, 
                w=self.loader.w, 
                focal=self.loader.focal, 
                near=1, 
                rays_o=self.rays_ori,
                rays_d=self.rays_dir
            )

    def __len__(self):
        return self.rays_ori.shape[0]
    
    def __getitem__(self, idx):
        rays_ori = self.rays_ori[idx]  # [3] or [B, 3]
        rays_dir = self.rays_dir[idx]  # [3] or [B, 3]
        rays_color = self.rays_color[idx]  # [3] or [B, 3]

        if self.split != 'train':
            rays_ori = rays_ori.reshape([-1,3])
            rays_dir = rays_dir.reshape([-1,3])
            rays_color = rays_color.reshape([-1,3])

        if self.loader.ndc:
            ndc_rays_ori = self.ndc_rays_ori[idx]
            ndc_rays_dir = self.ndc_rays_dir[idx]
            if self.split != 'train':
                ndc_rays_ori = ndc_rays_ori.reshape([-1,3])
                ndc_rays_dir = ndc_rays_dir.reshape([-1,3])
            return {'rays_ori': rays_ori, 'rays_dir': rays_dir, 
                    'rays_color': rays_color, 'ndc_rays_ori': ndc_rays_ori, 
                    'ndc_rays_dir': ndc_rays_dir,}

        return {'rays_ori': rays_ori, 'rays_dir': rays_dir, 
                'rays_color': rays_color, 'ndc_rays_ori': 0, 
                'ndc_rays_dir': 0,}


from .synthetic_dataset import convert_rays_to_sphere_coord
from scipy import spatial
from .utils import collect_rays
import random

@DATASETS.register_module()
class NeRFSphereDataset(Dataset):
    def __init__(self, size, loader, split, holdout=8):
        super().__init__()
        assert split in ('train', 'val', 'test'), split
        self.loader = build_from_cfg(loader, LOADERS)
        self.w, self.h = self.loader.w, self.loader.h
        self.size = size
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

        self.test_poses = np.stack( [pose for i, pose in enumerate(self.loader.poses) if i % holdout == 0], axis=0).astype('float32')
        # # patch sample
        # patch_row_split_num = self.h//self.size
        # patch_col_split_num = self.w//self.size
        # split_idx = []
        # for i in range(patch_row_split_num):
        #     for j in range(patch_col_split_num):
        #         split_idx.append([i*self.size, j*self.size])
        # if self.h%self.size > 0:
        #     for j in range(patch_col_split_num):
        #         split_idx.append([self.h-self.size, j])
        # if self.w%self.size > 0:
        #     for i in range(patch_row_split_num):
        #         split_idx.append([i, self.w-self.size])
        # if self.h%self.size > 0 and self.w%self.size > 0:
        #     split_idx.append([self.h-self.size, self.w-self.size])
        # self.split_idx = split_idx

    def __len__(self):
        return self.rays_ori.shape[0]
        # return self.rays_ori.shape[0]*len(self.split_idx)

    def __getitem__(self, idx):
        rays_ori = self.rays_ori[idx].reshape([-1,3])
        rays_dir = self.rays_dir[idx].reshape([-1,3])
        rays_color = self.rays_color[idx].reshape([-1,3])

        pose = self.poses[idx, :3, :4]
        h, w, focal = self.poses[0, :3, -1]
        aug_axis_degree = [(random.random()-0.5)*5 for _ in range(3)]
        aug_mat = spatial.transform.Rotation.from_euler('zyx', aug_axis_degree, degrees=True)
        # aug_axis_degree = random.random()*5+3
        # aug_mat = spatial.transform.Rotation.from_euler('z', aug_axis_degree, degrees=True)
        aug_mat = aug_mat.as_matrix()
        aug_pose = aug_mat @ pose

        aug_pose = self.test_poses[idx % len(self.test_poses), :3, :4]
        aug_rays_ori, aug_rays_dir = collect_rays(h, w, focal, aug_pose)
        aug_rays_ori = aug_rays_ori.reshape([-1,3])
        aug_rays_dir = aug_rays_dir.reshape([-1,3])

        rays_ori, rays_dir = torch.tensor(rays_ori).float(), torch.tensor(rays_dir).float()
        aug_rays_ori, aug_rays_dir = torch.tensor(aug_rays_ori).float(), torch.tensor(aug_rays_dir).float()

        # rand_interpolation = self.poses[int(random.random()*self.rays_ori.shape[0]), :, 3]
        # t = random.random()
        # aug_rays_ori = aug_rays_ori*t + rand_interpolation*(1-t)

        return {'uv': convert_rays_to_sphere_coord(rays_ori, rays_dir, self.loader.far),
                'st': convert_rays_to_sphere_coord(rays_ori, rays_dir, self.loader.far*1.5),
                'aug_uv': convert_rays_to_sphere_coord(aug_rays_ori, aug_rays_dir, self.loader.far),
                'aug_st': convert_rays_to_sphere_coord(aug_rays_ori, aug_rays_dir, self.loader.far*1.5),
                'rays_color': rays_color, 
                'h': self.h, 'w': self.w}


