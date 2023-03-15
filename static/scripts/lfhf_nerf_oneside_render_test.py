import subprocess
import os
import sys
sys.path.append(os.path.expanduser('~/code/nelf/'))
import json
import torch
from mmcv import Config
import matplotlib.pyplot as plt

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

root = './data/out'
out = './data/out/render-test'
if not os.path.isdir(out):
    os.mkdir(out)

scene_img = {
    'lego': ['r_8.png', 'r_48.png', 'r_108.png', 'r_198.png'],
}

for scene in scene_img.keys():
    path = f'{root}/oneside-{scene}-hfnerf'
    for f in os.listdir(path):
        if '.py' in f:
            print(f'{f} loaded.')
            break
    cfg_file = f'{path}/{f}'
    cfg = Config.fromfile(cfg_file)
    cfg.data.train.dataset.batch_size = -1
    cfg.data.val.split = 'test'
    cfg.data.val.testskip = 0
    cfg.data.val.select_imgs = scene_img[scene]
    dataset = build_dataset(cfg.data.val)
    model = build_renderer(cfg.model).cuda()
    state_dict = torch.load(f'{path}/best.pth')
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    if 'module' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # state_dict = new_state_dict
    model.load_state_dict(state_dict)
    cfg.evaluation.render_params.max_rays_num = 512
    h, w = dataset.h, dataset.w

    model.eval()
    num_images = len(dataset)
    for i in range(num_images):
        data = {}
        for k, v in dataset[i].items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda()
            else:
                data[k] = v
        
        with torch.no_grad():
            result = model.forward_render(**data, **cfg.evaluation.render_params)

        nelf_im = result['nelf_color_map'].reshape([h,w,-1]).cpu().numpy().clip(0,1)
        nerf_im = result['nerf_color_map'].reshape([h,w,-1]).cpu().numpy().clip(0,1)
        max_occ = result['nelf_max_occ_map'].reshape([h,w,-1]).cpu().numpy()
        nerf_max_occ = result['nerf_max_occ_map'].reshape([h,w,-1]).cpu().numpy()
        nerf_depth = result['nerf_depth_map'].reshape([h,w,-1]).cpu().numpy()

        nerf_aug = result.get('nerf_aug_color_map', result['color_map']).reshape([h,w,-1]).cpu().numpy()
        nelf_aug = result.get('nelf_aug_color_map', result['color_map']).reshape([h,w,-1]).cpu().numpy()
        gt = data['rays_color'].cpu().numpy().reshape([h,w,-1]) # extra dim
        plt.imsave(f'{out}/{scene}-{scene_img[scene][i]}-hf-nerf.png', nerf_im)
        plt.imsave(f'{out}/{scene}-{scene_img[scene][i]}-hf-depth.png', nerf_depth[:,:,0], cmap='Oranges') 


for scene in scene_img.keys():
    path = f'{root}/oneside-{scene}-lfnerf'
    for f in os.listdir(path):
        if '.py' in f:
            print(f'{f} loaded.')
            break
    cfg_file = f'{path}/{f}'
    cfg = Config.fromfile(cfg_file)
    cfg.data.train.dataset.batch_size = -1
    cfg.data.val.split = 'test'
    cfg.data.val.testskip = 0
    cfg.data.val.select_imgs = scene_img[scene]
    dataset = build_dataset(cfg.data.val)
    model = build_renderer(cfg.model).cuda()
    state_dict = torch.load(f'{path}/best.pth')
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    if 'module' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # state_dict = new_state_dict
    model.load_state_dict(state_dict)
    cfg.evaluation.render_params.max_rays_num = 512
    h, w = dataset.h, dataset.w

    model.eval()
    num_images = len(dataset)
    for i in range(num_images):
        data = {}
        for k, v in dataset[i].items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda()
            else:
                data[k] = v
        
        with torch.no_grad():
            result = model.forward_render(**data, **cfg.evaluation.render_params)

        nelf_im = result['nelf_color_map'].reshape([h,w,-1]).cpu().numpy().clip(0,1)
        nerf_im = result['nerf_color_map'].reshape([h,w,-1]).cpu().numpy().clip(0,1)
        max_occ = result['nelf_max_occ_map'].reshape([h,w,-1]).cpu().numpy()
        nerf_max_occ = result['nerf_max_occ_map'].reshape([h,w,-1]).cpu().numpy()
        nerf_depth = result['nerf_depth_map'].reshape([h,w,-1]).cpu().numpy()

        nerf_aug = result.get('nerf_aug_color_map', result['color_map']).reshape([h,w,-1]).cpu().numpy()
        nelf_aug = result.get('nelf_aug_color_map', result['color_map']).reshape([h,w,-1]).cpu().numpy()
        gt = data['rays_color'].cpu().numpy().reshape([h,w,-1]) # extra dim
        plt.imsave(f'{out}/{scene}-{scene_img[scene][i]}-lf-nerf.png', nerf_im)
        plt.imsave(f'{out}/{scene}-{scene_img[scene][i]}-lf-depth.png', nerf_depth[:,:,0], cmap='Oranges') 


