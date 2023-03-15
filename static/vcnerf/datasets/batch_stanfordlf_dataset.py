import os
import os.path as osp
import random
from numpy.lib.function_base import select
from scipy import spatial
import json
import glob
import numpy as np
from PIL import Image
import torch

from vcnerf.utils import get_root_logger
from .builder import DATASETS


@DATASETS.register_module()
class BatchStanfordLFDataset(object):
    def __init__(self, 
                 base_dir, 
                 downsample,
                 batch_size,
                 split,
                 testskip=8,
                 keep_idx=[],
                 given_uv=None,
                 scale=1024,
                 non_zero_pretrain=-1,
                 perturb=False,
                 to_cuda=True):
        super().__init__()
        self.logger = get_root_logger()
        self.base_dir = os.path.expanduser(base_dir)

        all_uv = []
        all_st = []
        all_color = []
        all_name  = []
        # image_paths = glob.glob(f"{self.base_dir}/sparse/*.png")
        image_paths = glob.glob(f"{self.base_dir}/*.png")
        image_paths.sort()
        if keep_idx:
            if isinstance(keep_idx[0], int):
                sel = lambda x: x in keep_idx
            elif isinstance(keep_idx[0], str):
                all_idx = []
                for idx, i in enumerate(image_paths):
                    for keep in keep_idx:
                        if keep in i:
                            all_idx.append(idx)
                            break
                sel = lambda x: x in all_idx
        else:
            sel = (lambda x: x%testskip!=0) if split=='train' else (lambda x: x%testskip==0)
        image_paths = [i for idx, i in enumerate(image_paths) if sel(idx)]

        for p in image_paths:
            all_name.append(p.split('/')[-1])
            img = Image.open(p)
            w, h = img.size
            img = img.resize([w//downsample, h//downsample], Image.BILINEAR)
            all_color.append(np.asarray(img)/255.0)
            uv = np.asarray([p.split('_')[3], p.split('_')[4].replace('.png','')], dtype='float32')
            xs = np.arange(0, h//downsample)
            ys = np.arange(0, w//downsample)
            s, t = np.meshgrid(xs, ys, indexing="ij")
            self.st_base = np.stack([s, t], -1)*downsample
            all_uv.append(uv)
        self.w, self.h = w//downsample, h//downsample
        self.downsample = downsample
        self.batch_size = batch_size
        self.scale = scale

        # self.all_uv = torch.from_numpy(np.stack(
        #     [j for i,j in enumerate(all_uv) if sel(i)], axis=0)).float()
        # self.all_st = torch.from_numpy(np.stack(
        #     [j for i,j in enumerate(all_st) if sel(i)], axis=0)).float()
        # self.all_color = torch.from_numpy(np.stack(
        #     [j for i,j in enumerate(all_color) if sel(i)], axis=0)).float()
        # self.all_name = [j for i,j in enumerate(all_name) if sel(i)]
        self.all_uv = torch.from_numpy(np.stack(all_uv, axis=0)).float()
        self.st_scale = max([self.w,self.h])
        self.uv_scale = self.all_uv.abs().max()
        if given_uv is not None:
            self.logger.info(f'overwrite uv from {self.all_uv} to')
            self.all_uv = torch.tensor(given_uv).float()*self.scale*2
            # self.all_uv = (self.all_uv+500)/1500*self.scale*2
            self.logger.info(f'{self.all_uv}')
        self.all_color = torch.from_numpy(np.stack(all_color, axis=0)).float()
        self.all_name = all_name
        self.st_base = torch.from_numpy(self.st_base).float()

        self.u_max, self.v_max = self.all_uv.view([-1,2]).max(0).values
        self.u_min, self.v_min = self.all_uv.view([-1,2]).min(0).values
        self.s_min, self.t_min = self.st_base.view([-1,2]).min(0).values
        self.s_max, self.t_max = self.st_base.view([-1,2]).max(0).values

        if to_cuda:
            self.all_uv = self.all_uv.cuda()
            self.all_color = self.all_color.cuda()
            self.st_base = self.st_base.cuda()
        self.center_uv = self.all_uv[torch.median(self.all_uv, dim=0).indices[0],:]/self.scale

        if perturb and not to_cuda:
            raise RuntimeError('perturb requires to cuda')
        self.perturb = perturb
        self.iter = 0
        self.non_zero_pretrain = non_zero_pretrain

    def __len__(self):
        return self.all_uv.shape[0]

    def __getitem__(self, idx):
        st = self.st_base
        rays_color = self.all_color[idx]
        grad_img_x = torch.mean(torch.abs(rays_color[:-1,:,:] - rays_color[1:,:,:]), -1, keepdim=True)
        grad_img_x = torch.cat([grad_img_x, grad_img_x[:1]], 0)
        grad_img_y = torch.mean(torch.abs(rays_color[:,:-1,:] - rays_color[:,1:,:]), -1, keepdim=True)
        grad_img_y = torch.cat([grad_img_y, grad_img_y[:,:1]], 1)
        rays_grad = torch.cat([grad_img_x, grad_img_y], dim=-1)

        if self.perturb:
            new_st = st + torch.rand_like(st)*2-1
            new_st[:,:,0] = new_st[:,:,0].clamp(0, self.h*2)
            new_st[:,:,1] = new_st[:,:,1].clamp(0, self.w*2)
            new_st_grid_coord = torch.zeros_like(new_st)
            new_st_grid_coord[:,:,0] = (new_st[:,:,1]/(self.w*2)-0.5)*2
            new_st_grid_coord[:,:,1] = (new_st[:,:,0]/(self.h*2)-0.5)*2
            new_color = torch.nn.functional.grid_sample(
                rays_color.permute([2,0,1])[None], new_st_grid_coord[None],)
                # mode='nearest')
            rays_color = new_color[0].permute([1,2,0])
            st = new_st

        st = st.reshape([-1,2])
        rays_color = rays_color.reshape([-1,3])
        rays_grad = rays_grad.reshape([-1,2])
        uv = self.all_uv[idx].expand_as(st)

        if self.non_zero_pretrain == self.iter:
            self.logger.info(f'pretrain on non-zero pixels finished')
        if self.iter < self.non_zero_pretrain:
            nonzero_mask = rays_color.sum(-1)>60/255
            st = st[nonzero_mask]
            rays_color = rays_color[nonzero_mask]
            rays_grad = rays_grad[nonzero_mask]
            uv = uv[nonzero_mask]

        if self.batch_size > 0:
            n = st.shape[0]
            perm = torch.randperm(n)
            select_mask = torch.linspace(0,n-1,n)[perm].long()[:self.batch_size]
        else:
            select_mask = torch.ones_like(uv)[:,0].bool()

        nearest_uv = (self.all_uv-self.all_uv[idx][None]).abs().sum(1).sort().indices[1:9]
        nearest_uv = self.all_uv[nearest_uv]
        u_min, v_min = nearest_uv.min(0).values
        u_max, v_max = nearest_uv.max(0).values
        select_uv = [random.random()*(u_max-u_min)+u_min, random.random()*(v_max-v_min)+v_min]
        aug_uv = torch.tensor(select_uv, device=uv.device)
        aug_st = (self.st_base).reshape([-1,2])/self.scale
        aug_uv = aug_uv.expand_as(aug_st)/self.scale

        return {'uv': uv[select_mask]/self.scale,
                'st': st[select_mask]/self.scale,
                'aug_uv': aug_uv[select_mask],
                'aug_st': aug_st[select_mask],
                # 'all_uv': self.all_uv/self.scale,
                'h': self.h, 'w': self.w,
                'rays_color': rays_color[select_mask], 
                'rays_grad': rays_grad[select_mask],}

