import os
import os.path as osp
import PIL
from PIL import Image
import json
import numpy as np
from numpy.lib.polynomial import polysub
import torch
import random

from vcnerf.utils import get_root_logger
from .builder import DATASETS


def get_rays(image_size, K, rt):
    # K: [3,3]
    # rt: [3,4], c2w
    rt = rt[:3,:]
    rays_ori = rt[:3,-1]
    mat = torch.cat([rt, torch.tensor([0,0,0,1]).reshape([1,4])], dim=0)
    switch_xy = torch.zeros_like(mat)
    switch_xy[0,1] = 1
    switch_xy[1,0] = 1
    switch_xy[2,2] = 1
    switch_xy[3,3] = 1
    mat = mat@switch_xy
    h, w = image_size
    i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))
    # pytorch's meshgrid has indexing='ij'
    i = i.t().to(K.device) 
    j = j.t().to(K.device) 
    dirs = torch.stack(
        [(j-h*.5)/K[1,1], (i-w*.5)/K[0,0], 
         torch.ones_like(i), torch.ones_like(i)], -1).reshape([-1,4]) # [N,4]
    rays_ori = rays_ori + torch.zeros([dirs.shape[0], 1], device=mat.device) # [N,3]
    # Rotate ray directions from camera frame to the world frame
    rays_dir = torch.matmul(mat, dirs.permute(1,0)).permute(1,0)[:,:3]-rays_ori # [N,3]
    return rays_ori, torch.nn.functional.normalize(rays_dir, p=2, dim=-1)


def get_bbox_binary(mask, margin=5):
    h, w = mask.shape
    rows = mask.sum(1)
    if rows.sum()>0:
        row_min = max(rows.nonzero()[0].min()-margin, 0)
        row_max = min(rows.nonzero()[0].max()+margin, h)
        cols = mask.sum(0)
        col_min = max(cols.nonzero()[0].min()-margin, 0)
        col_max = min(cols.nonzero()[0].max()+margin, w)
    else:
        row_min, row_max, col_min, col_max = 0,0,0,0
    bbox = (row_min, row_max, col_min, col_max)
    bbox_mask = np.zeros_like(mask)
    bbox_mask[row_min:row_max, col_min:col_max] = 1
    return bbox, bbox_mask


def add_img_with_cam(imgs, cams):
    new_img = []
    for img in imgs:
        for cam in cams:
            if f'cam{cam}' in img:
                new_img.append(img)
                break
    return new_img


@DATASETS.register_module()
class PhysicsDataset(object):
    def __init__(self, 
                 image_root, 
                 image_list, 
                 image_size, 
                 sample_points, 
                 total_frames, 
                 init_iters, 
                 rand_in_sample=1,
                 background='white'):
        """synthetic modified pedestrian detection SynMPD
        """
        super().__init__()
        self.logger = get_root_logger()
        self.image_root = os.path.expanduser(image_root)
        self.image_list = os.path.expanduser(image_list)
        with open(self.image_list, 'r') as f:
            self.imgs = f.read().split('\n')
        # self.imgs = add_img_with_cam(self.imgs, ['00','03','06','07','08','10','12'])
        # self.imgs = [i for idx, i in enumerate(self.imgs) if 'frame0000' in i]
        self.static_subset = [idx for idx, i in enumerate(self.imgs) if 'frame0000' in i]
        self.h, self.w = image_size
        self.image_size = image_size
        self.sample_points = sample_points
        self.near = 0.1
        self.far = 2
        self.iter = 0
        self.total_frames = total_frames
        self.init_iters = init_iters
        self.rand_in_sample = rand_in_sample
        self.background = background

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.iter < self.init_iters:
            idx = self.static_subset[idx%len(self.static_subset)]
        if self.iter == self.init_iters:
            print('init finished')

        img_path, cam_path, timestamp = self.imgs[idx].split(' ')
        # print(img_path)
        img = Image.open(osp.join(self.image_root, 'images', img_path))
        w, h  = img.size
        img = img.resize([self.w, self.h])

        img = np.asarray(img)/255.0
        empty = img[:,:,-1]>0
        img = img[:,:,:3]
        if self.background:
            img[~empty,:] = 1
        else:
            img[~empty,:] = 0
        bbox, bbox_mask = get_bbox_binary(empty)
        if self.rand_in_sample<1 and self.sample_points>0:
            in_samples = min(int(self.sample_points*self.rand_in_sample), bbox_mask.sum())
            out_samples = self.sample_points-in_samples
        else:
            in_samples = self.sample_points
            out_samples = 0

        with open(osp.join(self.image_root, 'cameras', cam_path), 'r') as f:
            cam_param = json.load(f)
        cam_param['K'] = np.asarray(cam_param['K'])
        c2w = np.linalg.inv(np.asarray(cam_param['RT']+[[0,0,0,1]])).copy() 
        assert cam_param['K'][0,2]/w==0.5
        cam_param['K'][0,:] *= self.w/w
        cam_param['K'][1,:] *= self.h/h

        if self.sample_points > 0:
            non_zero_idx = bbox_mask.reshape([-1,1]).nonzero()[0]
            zero_idx = (bbox_mask==0).reshape([-1,1]).nonzero()[0]
            select_mask = torch.zeros([self.h*self.w])
            if in_samples<len(non_zero_idx):
                non_zero = non_zero_idx.tolist()
                random.shuffle(non_zero)
                select_mask[non_zero[:in_samples]] = 1
            if out_samples>0:
                idx = random.sample(zero_idx.tolist(), out_samples)
                select_mask[idx] = 1
            select_mask = select_mask.reshape([self.h, self.w]).bool()
        else:
            select_mask = torch.ones([self.h, self.w]).bool()

        rays_ori, rays_dir = get_rays(self.image_size, 
                                      torch.tensor(cam_param['K']).float(), 
                                      torch.tensor(c2w).float())
        return {'rays_ori': rays_ori[select_mask.view(-1),:].view([-1,3]), 
                'rays_dir': rays_dir[select_mask.view(-1),:].view([-1,3]), 
                'rays_color': torch.tensor(img).float()[select_mask].view([-1,3]), 
                'timestamp': int(timestamp)/self.total_frames,
                'near': self.near, 
                'far': self.far}
