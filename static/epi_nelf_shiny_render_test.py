import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmcv import Config
import imageio
import argparse
import time

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-f", default="./data/out/")
args = parser.parse_args()

base = os.path.join(args.f)
for f in os.listdir(base):
    if '.py' in f:
        print(f'{f} loaded.')
        break
cfg_file = f'{base}/{f}'
state_dict = f'{base}/latest.pth'

cfg = Config.fromfile(cfg_file)
# cfg.data.val.llff_data_param = {'N_views': 27*2}
# cfg.data.val.llff_data_param = {'percentile': 38, 'N_views': 27*3, 'zrate': 0.8}
dataset = build_dataset(cfg.data.val)
dataset.orbiter_dataset.imgs = []
for i, ind in enumerate(dataset.orbiter_dataset.sfm.render_poses):
    img = dataset.orbiter_dataset.sfm.render_poses[ind]
    img['path'] = dataset.orbiter_dataset.sfm.imgs[0]['path']
    dataset.orbiter_dataset.imgs.append(img)
dataset.valid_idx = list(range(len(dataset.orbiter_dataset.imgs)))
model = build_renderer(cfg.model).cuda()
model.load_state_dict(torch.load(state_dict)['state_dict'])
h = dataset.h
w = dataset.w

all_epi_im = []
all_nelf_im = []
all_time = []
num_images = len(dataset)
model.eval()
for i in range(num_images):
    data = {}
    for k, v in dataset[i].items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda()[None]
        else:
            data[k] = v
    
    with torch.no_grad():
        s = time.time()
        result = model.forward_render(**data, **cfg.evaluation.render_params)
        all_time.append(time.time()-s)
    epi_im = result['epi_color_map'].clamp(0,1).cpu().numpy().reshape([h,w,3])
    nelf_im = result['nelf_color_map'].clamp(0,1).cpu().numpy().reshape([h,w,3])
    plt.imsave(f'{base}/pred-epi-{i:03d}.png', epi_im)
    plt.imsave(f'{base}/pred-nelf-{i:03d}.png', nelf_im)
    all_epi_im.append(epi_im)
    all_nelf_im.append(nelf_im)
    print(f'[{i:03d}]/[{num_images:03d}] image finished')

imageio.mimsave(f'{base}/pred-epi.gif', all_epi_im)
imageio.mimsave(f'{base}/pred-nelf.gif', all_nelf_im)
os.system(f'ffmpeg -y -framerate 8 -i {base}/pred-nelf-%03d.png -b 20M {base}/pred-nelf.avi')
os.system(f'ffmpeg -y -framerate 8 -i {base}/pred-epi-%03d.png -b 20M {base}/pred-epi.avi')

all_time = np.array(all_time)
print(f'mean time {all_time.mean()} ({all_time.std()})')
import ipdb; ipdb.set_trace()


