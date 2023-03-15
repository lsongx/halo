import os
from sys import maxunicode
import torch
import numpy as np
from mmcv import Config
import matplotlib.pyplot as plt
import imageio
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

# state_dict = './data/out/epoch_126.pth'

cfg = Config.fromfile(cfg_file)
cfg.data.train.dataset.batch_size=-1
# cfg.data.train.dataset.sample_points=1024
dataset = build_dataset(cfg.data.train).dataset
dataset_val = build_dataset(cfg.data.val)
# dataset = build_dataset(cfg.data.val)
model = build_renderer(cfg.model).cuda()
model.eval()
model.load_state_dict(torch.load(state_dict)['state_dict'])
# cfg.evaluation.render_params.single_ray_forward = True

h, w = dataset.h, dataset.w

def save(i):
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

    im = result['nelf_color_map'].reshape([h,w,3]).cpu().numpy()
    occ_im = result['epi_color_map'].reshape([h,w,3]).cpu().numpy()
    # aug_im = result.get('aug_color_map', result['nelf_color_map']).reshape([h,w,3]).cpu().numpy()
    nelf_epi = result['nelf_epi_map'].reshape([h,w]).cpu().numpy()
    epi_map = result.get('epi_map', result['epi_map']).reshape([h,w]).cpu().numpy()
    # gt = data['rays_color'].cpu().numpy().reshape([h,w,3])
    gt = data['rays_color'][0].cpu().numpy().reshape([h,w,3]) # extra dim
    fig, axes = plt.subplots(2, 3, figsize=(12,8), dpi=300)
    axes[0,0].imshow(gt)
    axes[0,1].imshow(im)
    axes[0,2].imshow(occ_im)
    im0 = axes[1,1].imshow(epi_map)
    fig.colorbar(im0, ax=axes[1,1])
    im1= axes[1,0].imshow(nelf_epi, cmap='inferno')
    fig.colorbar(im1, ax=axes[1,0])
    fig.savefig(f'{base}/tmp{i}.png', format='png')
    print(f'{base}/tmp{i}.png saved')
    plt.close('all')

def save_val():
    for i in range(len(dataset_val)):
        data_block = dataset_val[i]
        data = {}
        for k, v in data_block.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda()[None] # extra dim
            else:
                data[k] = torch.tensor(v).cuda()[None] # extra dim

        with torch.no_grad():
            result = model.forward_render(**data, **cfg.evaluation.render_params)
            loss = (result['nelf_color_map']-data['rays_color'].view([-1,3]))**2
            print(f'loss {loss.mean().item()}')

        im = result['nelf_color_map'].reshape([h,w,3]).cpu().numpy()
        occ_im = result['epi_color_map'].reshape([h,w,3]).cpu().numpy()
        # im = result['epi_color_map'].reshape([h,w,3]).cpu().numpy()
        # aug_im = result.get('aug_color_map', result['nelf_color_map']).reshape([h,w,3]).cpu().numpy()
        nelf_epi = result['nelf_epi_map'].reshape([h,w]).cpu().numpy()
        epi_map = result.get('epi_map', result['epi_map']).reshape([h,w]).cpu().numpy()
        gt = data['rays_color'][0].cpu().numpy().reshape([h,w,3]) # extra dim
        fig, axes = plt.subplots(2, 3, figsize=(12,8), dpi=300)
        axes[0,0].imshow(gt); axes[0,0].set_title('gt')
        axes[0,1].imshow(im); axes[0,1].set_title('im')
        axes[0,2].imshow(occ_im); axes[0,2].set_title('occ_im')
        im0 = axes[1,1].imshow(epi_map); axes[1,1].set_title('epi_map')
        fig.colorbar(im0, ax=axes[1,1])
        im1= axes[1,0].imshow(nelf_epi, cmap='inferno'); axes[1,0].set_title('nelf_epi')
        fig.colorbar(im1, ax=axes[1,0])
        fig.savefig(f'{base}/tmp{i}-val.png', format='png')
        print(f'{base}/tmp{i}-val.png saved')
        plt.close('all')

def gen_video(frames, without_epi=True):
    min_u, min_v = dataset.all_uv.min(0).values
    max_u, max_v = dataset.all_uv.max(0).values
    u_seq = torch.cat([
        torch.linspace(min_u, max_u, frames),
        torch.linspace(max_u, min_u, frames),
        torch.linspace(min_u, min_u, frames),
    ], dim=0)
    v_seq = torch.cat([
        torch.linspace(min_v, max_v, frames),
        torch.linspace(max_v, max_v, frames),
        torch.linspace(max_v, min_v, frames),
    ], dim=0)
    images, epis = [], []
    occ_images = []
    epi_max, epi_min = -1e5, 1e5
    for i, (u, v) in enumerate(zip(u_seq, v_seq)):
        data = {}
        uv = torch.tensor([u,v]).to(dataset.st_base.device).expand_as(dataset.st_base)
        data['uv'] = uv/dataset.scale
        # data['st'] = (dataset.st_base + uv)/dataset.scale
        data['st'] = (dataset.st_base)/dataset.scale
        data['aug_uv'] = data['uv']
        data['aug_st'] = data['st']
        data['h'] = h
        data['w'] = w
        data['rays_color'] = None
        with torch.no_grad():
            result = model.forward_render(**data, **cfg.evaluation.render_params, without_epi=without_epi)
            # result = model.forward_render(**data, **cfg.evaluation.render_params,)
        im = result['nelf_color_map'].reshape([h,w,3]).cpu().numpy()*255
        if not without_epi:
            occ_im = result['epi_color_map'].reshape([h,w,3]).cpu().numpy()*255
        epi = result['epi_map'].reshape([h,w]).cpu().numpy()
        epi_max = max(epi.max(), epi_max)
        epi_min = min(epi.min(), epi_min)
        im = im.astype('uint8')
        plt.imsave(f'{base}/frame{i:04d}.png', im)
        plt.imsave(f'{base}/frame{i:04d}_epi.png', epi)
        print(f'{i}/{len(u_seq)} saved')
        images.append(im)
        epis.append(epi)
        if not without_epi:
            occ_images.append(occ_im)
    imageio.mimsave(f'{base}/out-img.gif', images)
    if not without_epi:
        imageio.mimsave(f'{base}/occ-out-img.gif', occ_images)

    colormap = plt.get_cmap('inferno')
    for idx, epi in enumerate(epis):
        epi = (epi-epi_min)/(epi_max-epi_min)
        epis[idx] = (colormap(epi) * 255).astype(np.uint8)[:,:,:3]
    imageio.mimsave(f'{base}/out-nelf-epi.gif', epis)


# save(0)
# import ipdb; ipdb.set_trace()
# save_val()
gen_video(10, without_epi=False)
import ipdb; ipdb.set_trace()
num_images = len(dataset)
for i in range(num_images):
    save(i)


# import ipdb; ipdb.set_trace()
