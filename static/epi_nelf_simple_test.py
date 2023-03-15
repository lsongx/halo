import os
import torch
from mmcv import Config
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import argparse

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
    data_block = dataset[i]
    data = {}
    for k, v in data_block.items():
        if isinstance(v, torch.Tensor):
            # data[k] = v.cuda()
            data[k] = v.cuda()[None] # extra dim
        else:
            # data[k] = torch.tensor(v).cuda()
            data[k] = torch.tensor(v).cuda()[None] # extra dim

    with torch.no_grad():
        result = model.forward_render(**data, **cfg.evaluation.render_params)
        loss = (result['nelf_color_map']-data['rays_color'].view([-1,3]))**2
        print(f'loss {loss.mean().item()}')

    im = result['nelf_color_map'].reshape([h,w,-1]).cpu().clamp(0,1).numpy()
    occ_im = result['epi_color_map'].reshape([h,w,-1]).cpu().clamp(0,1).numpy()
    # aug_im = result.get('aug_color_map', result['nelf_color_map']).reshape([h,w,-1]).cpu().numpy()
    nelf_epi = result['nelf_epi_map'].reshape([h,w,-1]).cpu().numpy()
    epi_map = result.get('epi_map', result['epi_map']).reshape([h,w,-1]).cpu().numpy()
    # gt = data['rays_color'].cpu().numpy().reshape([h,w,-1])
    gt = data['rays_color'][0].cpu().numpy().reshape([h,w,-1]) # extra dim
    fig, axes = plt.subplots(2, 3, figsize=(12,8), dpi=300)
    axes[0,0].imshow(gt)
    axes[0,1].imshow(im)
    axes[0,2].imshow(occ_im)
    im0 = axes[1,1].imshow(epi_map)
    fig.colorbar(im0, ax=axes[1,1])
    im1= axes[1,0].imshow(nelf_epi, cmap='inferno')
    fig.colorbar(im1, ax=axes[1,0])
    fig.savefig(f'{base}/tmp{i}.png', format='png')
    plt.imsave(f'{base}/tmp{i}-nelf-im-raw.png', im)
    plt.imsave(f'{base}/tmp{i}-epi-im-raw.png', occ_im)
    plt.imsave(f'{base}/tmp{i}-epi-epi-raw.png', epi_map[...,0], vmin=model.near.item(), vmax=model.far.item())
    plt.imsave(f'{base}/tmp{i}-epi-nelf-raw.png', nelf_epi[...,0], vmin=model.near.item(), vmax=model.far.item())
    print(f'{base}/tmp{i}.png saved')
    plt.close('all')

    # def save_obj(points, colors):
    #     text = ''
    #     for p, c in zip(points, colors):
    #         text += f'v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} '+\
    #                 f'{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n'
    #     with open(f'{base}/tmp.obj', 'w') as f:
    #         f.write(text)
    # import numpy as np
    # K = np.array([[555.5, 0, 512], [0, 555.5, 512], [0, 0, 1]])
    # focal = 555.5
    # near = 2
    # far = 6
    # i, j = np.meshgrid(np.arange(512, dtype=np.float32), np.arange(512, dtype=np.float32), indexing='xy')
    # dirs = np.stack([(i-512*.5)/focal, -(j-512*.5)/focal, -np.ones_like(i)], -1).reshape([-1,3])
    # depth = (epi_map-epi_map.min())/(epi_map.max()-epi_map.min())*(far-near)+near
    # points = dirs*depth.reshape([-1,1])
    # # colors = np.zeros_like(dirs); colors[:,0] = 1
    # colors = gt.reshape([-1,3])
    # save_obj(points, colors)

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



