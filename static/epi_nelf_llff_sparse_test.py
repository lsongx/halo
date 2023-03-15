import os
import torch
from mmcv import Config
import matplotlib.pyplot as plt
import argparse
import imageio
import time
import numpy as np

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
cfg.data.train.dataset.batch_size = -1

dataset = build_dataset(cfg.data.train).dataset
num_images = 27*2
# spiral
# theta = torch.linspace(0, 5*3.1415926, num_images)
# points = torch.stack([theta*theta.sin(), theta*theta.cos()], dim=-1)/16+0.5
theta = torch.linspace(0, 2*3.1415926, num_images)
points = torch.stack([theta.sin(), theta.cos()], dim=-1)*0.8+0.5
all_poses = dataset.poses
new_poses = []
for p in points:
    pose = (all_poses[0]*(1-p[0])+all_poses[1]*p[0])*p[1] +\
           (all_poses[2]*(1-p[0])+all_poses[3]*p[0])*(1-p[1])
    new_poses.append(pose)
new_poses = torch.stack(new_poses, 0)
dataset.poses = new_poses
dataset.length = len(new_poses)

model = build_renderer(cfg.model).cuda()
model.load_state_dict(torch.load(state_dict)['state_dict'])
cfg.evaluation.render_params = {'max_rays_num': 512}

h, w = dataset.h, dataset.w

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

imageio.mimsave(f'{base}/pred-epi.gif', all_epi_im, fps=27)
imageio.mimsave(f'{base}/pred-nelf.gif', all_nelf_im, fps=27)
os.system(f'ffmpeg -y -framerate 8 -i {base}/pred-nelf-%03d.png -b 20M {base}/pred-nelf.avi')
os.system(f'ffmpeg -y -framerate 8 -i {base}/pred-epi-%03d.png -b 20M {base}/pred-epi.avi')

all_time = np.array(all_time)
print(f'mean time {all_time.mean()} ({all_time.std()})')
import ipdb; ipdb.set_trace()


