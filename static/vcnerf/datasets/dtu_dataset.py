import os
import math
import numpy as np
from PIL import Image
import torch

from vcnerf.utils import get_root_logger
from .builder import DATASETS
from .utils.dtu_loader import DVRDataset


def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
    )
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj


def gen_rays(poses, width, height, focal, c=None):
    """
    Generate camera rays
    :return (B, H, W, 8)
    """
    num_images = poses.shape[0]
    device = poses.device
    cam_unproj_map = (
        unproj_map(width, height, focal.squeeze(), c=c, device=device)
        .unsqueeze(0)
        .repeat(num_images, 1, 1, 1)
    )
    rays_ori = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
    rays_dir = torch.matmul(
        poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)
    )[:, :, :, :, 0]

    return rays_ori, rays_dir


@DATASETS.register_module()
class DTUDataset:
    def __init__(self, 
                 datadir, 
                 load_object,
                 batch_size, 
                 select_idx=None,
                 exclude_idx=None,
                 batching=True,
                 to_cuda=False,
                 llff_data_param={}):
        self.logger = get_root_logger()
        self.batch_size = batch_size
        datadir = os.path.expanduser(datadir)
        
        self.dvr_dataset = DVRDataset(datadir, load_object=load_object)
        data = self.dvr_dataset[0]
        if select_idx is not None:
            all_idx = torch.tensor(select_idx).long()
        else:
            all_idx = torch.tensor([k for k in range(len(data['poses'])) 
                                    if k not in exclude_idx]).long()

        self.near = self.dvr_dataset.z_near
        self.far = self.dvr_dataset.z_far
        self.logger.info(f'idx: {all_idx}')
        self.logger.info(f'NEAR {self.near} FAR {self.far}')

        self.imgs = data['images'][all_idx].permute([0,2,3,1])*0.5+0.5
        self.poses = data['poses'][all_idx]
        self.all_poses = data['poses']
        self.focal = data['focal'][None]
        self.c = data['c'][None]
        self.h, self.w = self.imgs.shape[1:3]
        self.render_poses = torch.tensor(self.dvr_dataset.render_poses)
        if to_cuda:
            self.imgs.cuda()
            self.poses.cuda()
            self.render_poses.cuda()
        self.batching = batching
        self.length = len(self.poses)
        if self.batching:
            self.logger.info(f'creating all rays')
            all_rays_ori, all_rays_dir = gen_rays(self.poses, self.w, self.h, self.focal, self.c) 
            self.logger.info(f'finish creating all rays')
            self.all_rays_ori = all_rays_ori.reshape([-1,3])
            self.all_rays_dir = all_rays_dir.reshape([-1,3])
            # self.imgs = self.imgs.view(-1,3)
            # self.length = self.imgs.shape[0]
            self.permute = torch.randperm(self.h*self.w*len(all_idx))
            self.all_rays_ori = self.all_rays_ori[self.permute]
            self.all_rays_dir = self.all_rays_dir[self.permute]
            self.all_rays_color = self.imgs.reshape([-1,3])[self.permute]
            self.length = math.ceil(self.all_rays_color.shape[0] / self.batch_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.batching:
            e = min(idx*self.batch_size+self.batch_size, 
                    self.all_rays_color.shape[0])
            s = e-self.batch_size
            return {'rays_ori': self.all_rays_ori[s:e],
                    'rays_dir': self.all_rays_dir[s:e],
                    'rays_color': self.all_rays_color[s:e],
                    'near': self.near, 'far': self.far}
            # return {'rays_ori': self.all_rays_ori[idx,:], 
            #         'rays_dir': self.all_rays_dir[idx,:], 
            #         'rays_color': self.imgs[idx,:], 
            #         'near': self.near, 'far': self.far}
        else:
            target = self.imgs[idx%len(self.imgs)]
            pose = self.poses[idx, :3,:4]
            rays_ori, rays_dir = gen_rays(pose[None], self.w, self.h, self.focal, self.c) 

        if self.batch_size == -1:
            return {'rays_ori': rays_ori.view([-1,3]), 
                    'rays_dir': rays_dir.view([-1,3]), 
                    'rays_color': target.view([-1,3]), 
                    'near': self.near, 'far': self.far}

        select_idx = torch.randperm(self.h*self.w)[:self.batch_size]
        rays_ori = rays_ori.view([-1,3])[select_idx]  # (N, 3)
        rays_dir = rays_dir.view([-1,3])[select_idx]  # (N, 3)
        rays_color = target.view([-1,3])[select_idx]  # (N, 3)

        return {'rays_ori': rays_ori, 'rays_dir': rays_dir, 
                'rays_color': rays_color, 'near': self.near, 'far': self.far}


