import os
from sys import maxunicode
import torch
from mmcv import Config
import matplotlib.pyplot as plt
import imageio

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

# cfg_file = './configs/nelf_synthetic_lego.py'
cfg_file = './configs/nelf_llff_fern.py'
# cfg_file = './configs/nelf_llff_fern_adv.py'
cfg_file = './configs/nelf_stanfordlf.py'
# cfg_file = './configs/adv_nelf_stanfordlf.py'
# cfg_file = './configs/style_nelf_stanfordlf.py'
cfg_file = './configs/fast_nelf_stanfordlf.py'
state_dict = './data/out/latest.pth'
# state_dict = './data/out/checkpoint_12.22.pth'

cfg = Config.fromfile(cfg_file)
# cfg.data.train.dataset.sample_points=-1
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
        loss = (result['color_map']-data['rays_color'].view([-1,3]))**2
        print(f'loss {loss.mean().item()}')

    im = result['color_map'].reshape([h,w,3]).cpu().numpy()
    aug_im = result.get('aug_color_map', result['color_map']).reshape([h,w,3]).cpu().numpy()
    # gt = data['rays_color'].cpu().numpy().reshape([h,w,3])
    gt = data['rays_color'][0].cpu().numpy().reshape([h,w,3]) # extra dim
    print(f'diff {((im-gt)**2).mean()}')
    print(f'aug diff {((aug_im-gt)**2).mean()}')
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
    axes[0,0].imshow(gt)
    axes[0,1].imshow(im)
    axes[1,1].imshow(aug_im)
    fig.savefig(f'./data/out/tmp{i}.png', format='png')
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
            loss = (result['color_map']-data['rays_color'].view([-1,3]))**2
            print(f'loss {loss.mean().item()}')

        im = result['color_map'].reshape([h,w,3]).cpu().numpy()
        aug_im = result.get('aug_color_map', result['color_map']).reshape([h,w,3]).cpu().numpy()
        gt = data['rays_color'][0].cpu().numpy().reshape([h,w,3]) # extra dim
        print(f'diff {((im-gt)**2).mean()}')
        print(f'aug diff {((aug_im-gt)**2).mean()}')
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
        axes[0,0].imshow(gt)
        axes[0,1].imshow(im)
        axes[1,1].imshow(aug_im)
        fig.savefig(f'./data/out/tmp{i}-val.png', format='png')
        plt.close('all')

def gen_video(frames):
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
    images = []
    for i, (u, v) in enumerate(zip(u_seq, v_seq)):
        data = {}
        uv = torch.tensor([u,v]).to(dataset.st_base.device).expand_as(dataset.st_base)
        data['uv'] = uv/dataset.scale
        data['st'] = (dataset.st_base + uv)/dataset.scale
        data['aug_uv'] = data['uv']
        data['aug_st'] = data['st']
        data['h'] = h
        data['w'] = w
        data['rays_color'] = None
        with torch.no_grad():
            result = model.forward_render(**data, **cfg.evaluation.render_params)
        im = result['color_map'].reshape([h,w,3]).cpu().numpy()*255
        im = im.astype('uint8')
        plt.imsave(f'./data/out/frame{i:04d}.png', im)
        images.append(im)
    imageio.mimsave('./data/out/out.gif', images)


gen_video(10)
import ipdb; ipdb.set_trace()
num_images = len(dataset)
for i in range(num_images):
    save(i)
save_val()


# import ipdb; ipdb.set_trace()
