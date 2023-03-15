import torch
import numpy as np
import sys
import os
sys.path.append(os.path.expanduser('~/code/nelf'))
import json
import argparse
import imageio
import matplotlib.pyplot as plt
from mmcv import Config

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-f", type=str, default='./data/out/')
parser.add_argument("-k", type=int, default=0)
parser.add_argument("-g", type=int, default=-1)
parser.add_argument("-t", type=int, default=3)
args = parser.parse_args()
if args.g < 0:
    args.g = args.k
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.g}'


scenes = {
    'eucalyptus-flowers': "",
    'lego-bulldozer': "",
    'treasure': "",
    'amethyst': "",
    'bracelet': "",
    'jellybeans': "",
    'lego-truck': "",
    'stanfordbunny': "",
    'chess': "",
    'lego-knights': "",
    'cards-big': "",
    'cards-small': "",
    'lego-gantry': "",
}

while args.k < len(scenes):
    scene = list(scenes.keys())[args.k]

    base = os.path.expanduser(os.path.join(args.f, scene))
    for f in os.listdir(base):
        if '.py' in f:
            print(f'{base}/{f} loaded.')
            break
    cfg_file = f'{base}/{f}'
    state_dict = f'{base}/best.pth'

    if os.path.isdir(f'{base}/render-save'):
        os.system(f'rm -rf {base}/render-save/*')
    else:
        os.mkdir(f'{base}/render-save')

    cfg = Config.fromfile(cfg_file)
    model = build_renderer(cfg.model).cuda()
    state_dict = torch.load(state_dict)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    if 'module' in list(state_dict.keys())[0]:
        new = {}
        for k, v in state_dict.items():
            new[k.replace('module.', '')] = v
        state_dict = new
    model.load_state_dict(state_dict)
    model.eval()
    cfg.data.train.dataset.batch_size=-1
    dataset = build_dataset(cfg.data.train).dataset
    h, w = dataset.h, dataset.w

    frames = 27*4
    min_u, min_v = dataset.all_uv.min(0).values
    max_u, max_v = dataset.all_uv.max(0).values
    
    t = torch.linspace(0.5*3.1415926, 2.5*3.1415926, frames, device=min_u.device)
    coords = torch.stack([t.cos(), 2*t.sin()*t.cos()], -1)/2+0.5
    uv_seq = torch.stack([
        coords[:,0]*(max_u-min_u)+min_u,
        coords[:,1]*(max_v-min_v)+min_v,
    ], -1)
    
    images, epis = [], []
    occ_images = []
    epi_max, epi_min = -1e5, 1e5
    for i, uv in enumerate(uv_seq):
        data = {}
        uv = uv.to(dataset.st_base.device).expand_as(dataset.st_base)
        data['uv'] = uv/dataset.scale
        # data['st'] = (dataset.st_base + uv)/dataset.scale
        data['st'] = (dataset.st_base)/dataset.scale
        data['aug_uv'] = data['uv']
        data['aug_st'] = data['st']
        data['h'] = h
        data['w'] = w
        data['rays_color'] = None
        with torch.no_grad():
            result = model.forward_render(**data, **cfg.evaluation.render_params, without_epi=False)
        nelf_im = result['nelf_color_map'].reshape([h,w,3]).cpu().clamp(0,1).numpy()*255
        occ_im = result['epi_color_map'].reshape([h,w,3]).cpu().clamp(0,1).numpy()*255
        epi = result['epi_map'].reshape([h,w]).cpu().numpy()
        epi_max = max(epi.max(), epi_max)
        epi_min = min(epi.min(), epi_min)
        nelf_im = nelf_im.astype('uint8')
        occ_im = occ_im.astype('uint8')
        # plt.imsave(f'{base}/render-save/frame{i:04d}_nelf.png', nelf_im)
        plt.imsave(f'{base}/render-save/frame{i:04d}_epi.png', epi)
        plt.imsave(f'{base}/render-save/frame{i:04d}_im.png', occ_im)
        print(f'{i}/{len(uv_seq)} saved')
        # images.append(nelf_im)
        epis.append(epi)
        occ_images.append(occ_im)
    # imageio.mimsave(f'{base}/render-save/out-img.gif', images)
    imageio.mimsave(f'{base}/render-save/out-img.gif', occ_images)

    colormap = plt.get_cmap('inferno')
    for idx, epi in enumerate(epis):
        epi = (epi-epi_min)/(epi_max-epi_min)
        epis[idx] = (colormap(epi) * 255).astype(np.uint8)[:,:,:3]
    imageio.mimsave(f'{base}/render-save/out-epi.gif', epis)

    args.k += args.t
