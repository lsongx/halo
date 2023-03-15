import os
import torch
from mmcv import Config
import matplotlib.pyplot as plt
import argparse
import imageio
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

dataset = build_dataset(cfg.data.val)
dataset = build_dataset(cfg.data.train).dataset

model = build_renderer(cfg.model).cuda()
model.load_state_dict(torch.load(state_dict)['state_dict'])
cfg.evaluation.render_params.max_rays_num = 512
cfg.evaluation.render_params.z_by_nelf = 1.0
# cfg.evaluation.render_params.n_samples = 2

h, w = dataset.h, dataset.w

# model.eval()
model.train()
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

    nelf_im = result['nelf_color_map'].reshape([h,w,-1]).cpu().numpy()
    nerf_im = result['nerf_color_map'].reshape([h,w,-1]).cpu().numpy()
    nelf_max_occ = result['nelf_max_occ_map'].reshape([h,w,-1]).cpu().numpy()
    nerf_max_occ = result['nerf_max_occ_map'].reshape([h,w,-1]).cpu().numpy()

    none_img = result['nerf_max_occ_map']*0
    nerf_aug = result.get('nerf_aug_color_map', none_img).reshape([h,w,-1]).cpu().numpy()
    nelf_aug = result.get('nelf_aug_color_map', none_img).reshape([h,w,-1]).cpu().numpy()
    nerf_aug_occ = result.get('nerf_aug_max_occ', none_img).reshape([h,w,-1]).cpu().numpy()
    nelf_aug_occ = result.get('nelf_aug_max_occ', none_img).reshape([h,w,-1]).cpu().numpy()

    lf_nerf_color_map = result.get('lf_nerf_color_map', none_img).reshape([h,w,-1]).cpu().numpy()
    lf_nerf_acc_map = result.get('lf_nerf_acc_map', none_img).reshape([h,w,-1]).cpu().numpy()
    lf_nerf_depth_map = result.get('lf_nerf_depth_map', none_img).reshape([h,w,-1]).cpu().numpy()
    lf_nerf_empty_mask = result.get('lf_nerf_empty_mask', none_img).reshape([h,w,-1]).cpu().numpy()

    gt = data['rays_color'].cpu().numpy().reshape([h,w,-1]) # extra dim

    print(f'diff {((nelf_im-gt)**2).mean()}')
    # fig, axes = plt.subplots(1, 2, figsize=(8,4), dpi=300)
    fig, axes = plt.subplots(4, 4, figsize=(15,15), dpi=200)
    axes[0,0].imshow(gt); axes[0,0].set_title('gt')
    axes[0,1].imshow(nelf_im); axes[0,1].set_title('nelf_im')
    axes[1,0].imshow(nerf_im); axes[1,0].set_title('nerf_im')
    axes[1,1].imshow(nelf_max_occ); axes[1,1].set_title('nelf_max_occ')
    axes[2,0].imshow(nerf_max_occ); axes[2,0].set_title('nerf_max_occ')
    axes[0,2].imshow(nelf_aug); axes[0,2].set_title('nelf_aug')
    axes[1,2].imshow(nerf_aug); axes[1,2].set_title('nerf_aug')
    axes[2,1].imshow(nelf_aug_occ); axes[2,1].set_title('nelf_aug_occ')
    axes[2,2].imshow(nerf_aug_occ); axes[2,2].set_title('nerf_aug_occ')
    
    axes[3,0].imshow(lf_nerf_color_map); axes[3,0].set_title('lf_nerf_color_map')
    axes[3,1].imshow(lf_nerf_acc_map); axes[3,1].set_title('lf_nerf_acc_map')
    axes[3,2].imshow(lf_nerf_depth_map); axes[3,2].set_title('lf_nerf_depth_map')
    axes[3,3].imshow(lf_nerf_empty_mask); axes[3,3].set_title('lf_nerf_empty_mask')
    
    fig.savefig(f'{base}/tmp{i}.png', format='png')
    import ipdb; ipdb.set_trace()


# all_coarse_im = []
# all_fine_im = []
# num_images = len(dataset)
# for i in range(num_images):
#     data = {}
#     for k, v in dataset[i].items():
#         if isinstance(v, torch.Tensor):
#             data[k] = v.cuda()
#         else:
#             data[k] = v
    
#     with torch.no_grad():
#         result = model.forward_render(**data, **cfg.evaluation.render_params)
#     coarse_im = result['coarse']['color_map'].clamp(0,1).cpu().numpy().reshape([h,w,3])
#     fine_im = result['fine']['color_map'].clamp(0,1).cpu().numpy().reshape([h,w,3])
#     plt.imsave(f'./data/out/pred-coarse-{i:03d}.png', coarse_im)
#     plt.imsave(f'./data/out/pred-fine-{i:03d}.png', fine_im)
#     all_coarse_im.append(coarse_im)
#     all_fine_im.append(fine_im)
#     print(f'[{i:03d}]/[{num_images:03d}] image finished')

# imageio.mimsave('./data/out/pred-coarse.gif', all_coarse_im)
# imageio.mimsave('./data/out/pred-fine.gif', all_fine_im)
# os.system(f'ffmpeg -y -framerate 8 -i ./data/out/pred-fine-%03d.png -b 20M ./data/out/pred-fine.avi')

# import ipdb; ipdb.set_trace()



