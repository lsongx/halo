import os
import torch
import matplotlib.pyplot as plt
from mmcv import Config
import imageio
import argparse

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-f", default="./data/out/")
parser.add_argument("-c", default=None)
args = parser.parse_args()

base = os.path.join(args.f)
for f in os.listdir(base):
    if '.py' in f:
        print(f'{f} loaded.')
        break
cfg_file = f'{base}/{f}'
if args.c is None:
    args.c = f'{base}/latest.pth'
else:
    args.c = os.path.expanduser(args.c)

if os.path.isdir(f'{base}/render-save'):
    os.system(f'rm -rf {base}/render-save/*')
else:
    os.mkdir(f'{base}/render-save')

cfg = Config.fromfile(cfg_file)
cfg.data.val.num_render_img = 27*4
dataset = build_dataset(cfg.data.val)
dataset.poses = dataset.render_poses
dataset.length = len(dataset.poses)
model = build_renderer(cfg.model).cuda()
state_dict = torch.load(args.c)
if 'state_dict' in state_dict.keys():
    state_dict = state_dict['state_dict']
if 'module' in list(state_dict.keys())[0]:
    new = {}
    for k, v in state_dict.items():
        new[k.replace('module.', '')] = v
    state_dict = new
model.load_state_dict(state_dict)
cfg.evaluation.render_params.max_rays_num = 512
h = dataset.h
w = dataset.w

all_im = []
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
    im = result['color_map'].clamp(0,1).cpu().numpy().reshape([h,w,3])
    plt.imsave(f'{base}/render-save/pred-{i:03d}.png', im)
    all_im.append(im)
    print(f'[{i:03d}]/[{num_images:03d}] image finished')

imageio.mimsave(f'{base}/render-save/pred.gif', all_im)
os.system(f'ffmpeg -y -framerate 8 -i {base}/render-save/pred-%03d.png -b 20M {base}/render-save/pred.mp4')

# import ipdb; ipdb.set_trace()


