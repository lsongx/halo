import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmcv import Config
import imageio
import time

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

base = './data/out'
base = './data/out/noproj-basescale1'
for f in os.listdir(base):
    if 'py' in f:
        break
cfg_file = f'{base}/{f}'
state_dict = f'{base}/latest.pth'

cfg = Config.fromfile(cfg_file)
cfg.data.val.llff_data_param = {'N_views': 27*2}
# cfg.data.val.llff_data_param = {'percentile': 38, 'N_views': 27*3, 'zrate': 0.8}
dataset = build_dataset(cfg.data.val)
dataset.poses = dataset.render_poses
model = build_renderer(cfg.model).cuda()
model.load_state_dict(torch.load(state_dict)['state_dict'])
cfg.evaluation.render_params.max_rays_num = 512
cfg.evaluation.render_params.z_by_nelf = 0.2
cfg.evaluation.render_params.n_samples = 16
h = dataset.h
w = dataset.w

all_nerf_im = []
all_nelf_im = []
all_time = []
num_images = len(dataset)
model.eval()
for i in range(num_images):
    data = {}
    for k, v in dataset[i].items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda()
        else:
            data[k] = v
    
    with torch.no_grad():
        s = time.time()
        result = model.forward_render(**data, **cfg.evaluation.render_params)
        all_time.append(time.time()-s)
    nerf_im = result['nerf_color_map'].clamp(0,1).cpu().numpy().reshape([h,w,3])
    nelf_im = result['color_map'].clamp(0,1).cpu().numpy().reshape([h,w,3])
    plt.imsave(f'./data/out/pred-nerf-{i:03d}.png', nerf_im)
    plt.imsave(f'./data/out/pred-nelf-{i:03d}.png', nelf_im)
    all_nerf_im.append(nerf_im)
    all_nelf_im.append(nelf_im)
    print(f'[{i:03d}]/[{num_images:03d}] image finished')

imageio.mimsave('./data/out/pred-nerf.gif', all_nerf_im)
imageio.mimsave('./data/out/pred-nelf.gif', all_nelf_im)
os.system(f'ffmpeg -y -framerate 8 -i ./data/out/pred-nelf-%03d.png -b 20M ./data/out/pred-nelf.avi')
os.system(f'ffmpeg -y -framerate 8 -i ./data/out/pred-nerf-%03d.png -b 20M ./data/out/pred-nerf.avi')

all_time = np.array(all_time)
print(f'mean time {all_time.mean()} ({all_time.std()})')
import ipdb; ipdb.set_trace()


