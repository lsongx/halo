import os
import torch
from mmcv import Config
import matplotlib.pyplot as plt
import argparse
import imageio

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

dataset = build_dataset(cfg.data.val)
# dataset = build_dataset(cfg.data.train).dataset

model = build_renderer(cfg.model).cuda()
model.load_state_dict(torch.load(state_dict)['state_dict'])
cfg.evaluation.render_params = {'max_rays_num': 512}

h, w = dataset.h, dataset.w

model.eval()
num_images = len(dataset)
for i in range(num_images):
    data = {}
    for k, v in dataset[i].items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda()[None]
        else:
            data[k] = v

    with torch.no_grad():
        result = model.forward_render(**data, **cfg.evaluation.render_params)

    nelf_im = result['nelf_color_map'].reshape([h,w,3]).clamp(0,1).cpu().numpy()
    epi_im = result['epi_color_map'].reshape([h,w,3]).clamp(0,1).cpu().numpy()
    nelf_epi_map = result['nelf_epi_map'].reshape([h,w,1]).cpu().numpy()
    epi_map = result['epi_map'].reshape([h,w,1]).cpu().numpy()
    # nerf_aug = result['nerf_aug_color_map'].reshape([h,w,3]).cpu().numpy()
    # nelf_aug = result['nelf_aug_color_map'].reshape([h,w,3]).cpu().numpy()
    gt = data['rays_color'].cpu().numpy().reshape([h,w,3]) # extra dim
    print(f'diff {((nelf_im-gt)**2).mean()}')
    # fig, axes = plt.subplots(1, 2, figsize=(8,4), dpi=300)
    fig, axes = plt.subplots(3, 3, figsize=(12,12), dpi=300)
    axes[0,0].imshow(gt); axes[0,0].set_title('gt')
    axes[0,1].imshow(nelf_im); axes[0,1].set_title('nelf_im')
    axes[1,0].imshow(epi_im); axes[1,0].set_title('epi_im')
    axes[1,1].imshow(nelf_epi_map); axes[1,1].set_title('nelf_epi_map')
    axes[2,0].imshow(epi_map); axes[2,0].set_title('epi_map')
    # axes[0,2].imshow(nelf_aug); axes[0,2].set_title('nelf_aug')
    # axes[1,2].imshow(nerf_aug); axes[1,2].set_title('nerf_aug')
    fig.savefig(f'{base}/tmp{i}.png', format='png')

    plt.imsave(f'{base}/tmp{i}-nelf-im-raw.png', nelf_im)
    plt.imsave(f'{base}/tmp{i}-epi-im-raw.png', epi_im)
    vmax = max(epi_map.max(), nelf_epi_map.max())
    vmin = min(epi_map.min(), nelf_epi_map.min())
    plt.imsave(f'{base}/tmp{i}-epi-epi-raw.png', epi_map[...,0], vmin=0, vmax=1)
    plt.imsave(f'{base}/tmp{i}-epi-nelf-raw.png', nelf_epi_map[...,0], vmin=0, vmax=1)
    import ipdb; ipdb.set_trace()


all_coarse_im = []
all_fine_im = []
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
    coarse_im = result['coarse']['color_map'].clamp(0,1).cpu().numpy().reshape([h,w,3])
    fine_im = result['fine']['color_map'].clamp(0,1).cpu().numpy().reshape([h,w,3])
    plt.imsave(f'{base}/pred-coarse-{i:03d}.png', coarse_im)
    plt.imsave(f'{base}/pred-fine-{i:03d}.png', fine_im)
    all_coarse_im.append(coarse_im)
    all_fine_im.append(fine_im)
    print(f'[{i:03d}]/[{num_images:03d}] image finished')

imageio.mimsave(f'{base}/pred-coarse.gif', all_coarse_im)
imageio.mimsave(f'{base}/pred-fine.gif', all_fine_im)
os.system(f'ffmpeg -y -framerate 8 -i {base}/pred-fine-%03d.png -b 20M {base}/pred-fine.avi')

import ipdb; ipdb.set_trace()



