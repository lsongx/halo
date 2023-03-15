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
class StanfordLFDataset(object):
    def __init__(self, 
                 base_dir, 
                 downsample,
                 batch_size,
                 testskip,
                 split,
                 scale=1024,
                 to_cuda=True,
                 add_aug=True):
        super().__init__()
        self.logger = get_root_logger()
        self.base_dir = os.path.expanduser(base_dir)

        all_uv = []
        all_st = []
        all_color = []
        all_name  = []
        image_paths = glob.glob(f"{self.base_dir}/sparse/*.png")
        for p in image_paths:
            all_name.append(p.split('/')[-1])
            img = Image.open(p)
            w, h = img.size
            img = img.resize([w//downsample, h//downsample])
            all_color.append(np.asarray(img)/255.0)
            uv = np.asarray([p.split('_')[3], p.split('_')[4].replace('.png','')], dtype='float32')
            xs = np.arange(0, h//downsample)
            ys = np.arange(0, w//downsample)
            s, t = np.meshgrid(xs, ys, indexing="ij")
            st = np.stack([s, t], -1)*downsample+uv
            self.st_base = np.stack([s, t], -1)*downsample
            all_uv.append(uv)
            all_st.append(st)
        self.w, self.h = w//downsample, h//downsample
        self.downsample = downsample
        self.scale = scale

        sel = (lambda x: x%testskip!=0) if split=='train' else (lambda x: x%testskip==0)
        self.all_uv = torch.from_numpy(np.stack(
            [j for i,j in enumerate(all_uv) if sel(i+1)], axis=0)).float()
        self.all_st = torch.from_numpy(np.stack(
            [j for i,j in enumerate(all_st) if sel(i+1)], axis=0)).float()
        self.all_color = torch.from_numpy(np.stack(
            [j for i,j in enumerate(all_color) if sel(i+1)], axis=0)).float()
        self.all_name = [j for i,j in enumerate(all_name) if sel(i+1)]
        self.st_base = torch.from_numpy(self.st_base).float()

        # import matplotlib.pyplot as plt
        # def t(v):
        #     self.all_st = self.all_st/v
        #     fig,ax=plt.subplots(1,1,dpi=600)
        #     for i in range(10):
        #         ax.scatter(self.all_st[i][::10,::10,:].reshape([-1,2])[:,0],self.all_st[i][::10,::10,:].reshape([-1,2])[:,1],s=0.05)
        #     fig.savefig(f'./data/out/tmp{v}.png')
        # import ipdb;ipdb.set_trace()
        u_min, v_min = self.all_uv.view([-1,2]).min(0).values
        self.all_uv[:,0] -= u_min+512
        self.all_uv[:,1] -= v_min+512 # starts from 512
        self.u_max, self.v_max = self.all_uv.view([-1,2]).max(0).values
        self.u_min, self.v_min = self.all_uv.view([-1,2]).min(0).values
        self.s_max, self.t_max = self.all_st.view([-1,2]).max(0).values
        self.s_min, self.t_min = self.all_st.view([-1,2]).min(0).values

        if to_cuda:
            self.all_uv = self.all_uv.cuda()
            self.all_st = self.all_st.cuda()
            self.all_color = self.all_color.cuda()
            self.st_base = self.st_base.cuda()
        self.add_aug = add_aug

    def __len__(self):
        return self.all_uv.shape[0]

    def __getitem__(self, idx):
        st = self.all_st[idx].reshape([-1,2])
        uv = self.all_uv[idx].expand_as(st)
        rays_color = self.all_color[idx].reshape([-1,3])

        if self.add_aug:
            nearest_uv = (self.all_uv-self.all_uv[idx][None]).abs().sum(1).sort().indices[1:9]
            nearest_uv = self.all_uv[nearest_uv]
            u_min, v_min = nearest_uv.min(0).values
            u_max, v_max = nearest_uv.max(0).values
            select_uv = [random.random()*(u_max-u_min)+u_min, random.random()*(v_max-v_min)+v_min]
            aug_uv = torch.tensor(select_uv, device=uv.device)
            aug_st = (self.st_base+aug_uv).reshape([-1,2])/self.scale
            aug_uv = aug_uv.expand_as(aug_st)/self.scale
        else:
            aug_uv, aug_st = 0, 0

        return {'uv': uv/self.scale,
                'st': st/self.scale,
                'aug_uv': aug_uv,
                'aug_st': aug_st,
                'h': self.h, 'w': self.w,
                'rays_color': rays_color, }

