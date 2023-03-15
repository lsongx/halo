import subprocess
import torch
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-k", type=int, default=0)
parser.add_argument("-g", type=int, default=-1)
parser.add_argument("-t", type=int, default=3)
args = parser.parse_args()
if args.g < 0:
    args.g = args.k
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.g}'

scene = {    
    'lego': "['r_2.png', 'r_16.png', 'r_93.png', 'r_55.png', 'r_73.png', 'r_86.png', 'r_26.png', 'r_75.png']",
    'chair': "['r_86.png', 'r_73.png', 'r_26.png', 'r_2.png', 'r_55.png', 'r_93.png', 'r_16.png', 'r_75.png']",
    'drums': "['r_86.png', 'r_93.png', 'r_75.png', 'r_26.png', 'r_55.png', 'r_73.png', 'r_16.png', 'r_2.png']",
    'ficus': "['r_2.png', 'r_93.png', 'r_73.png', 'r_86.png', 'r_75.png', 'r_26.png', 'r_55.png', 'r_16.png']",
    'mic': "['r_55.png', 'r_2.png', 'r_93.png', 'r_75.png', 'r_16.png', 'r_26.png', 'r_86.png', 'r_73.png']",
    'ship': "['r_55.png', 'r_93.png', 'r_26.png', 'r_75.png', 'r_16.png', 'r_33.png', 'r_73.png', 'r_86.png']",
    'materials': "['r_75.png', 'r_26.png', 'r_93.png', 'r_55.png', 'r_86.png', 'r_16.png', 'r_2.png', 'r_73.png']",
    'hotdog': "['r_16.png', 'r_93.png', 'r_75.png', 'r_86.png', 'r_2.png', 'r_55.png', 'r_73.png', 'r_26.png']",
}


if __name__ == '__main__':
    while args.k < len(scene):
        k = list(scene.keys())[args.k]
        v = scene[k]
        subprocess.run(f"python -m torch.distributed.launch --nproc_per_node=1 train.py "
                       f"--seed 1 --use_fp16 1 --config ./configs/nelf_nerf_synthetic_lego_sparse_hf_fail.py "
                       f"--work_dir ./data/out/{k}-hf-fail --port {25666+args.k} "
                       f"--cfg-options data.train.dataset.base_dir='~/data/3d/nerf/nerf_synthetic/{k}' "
                       f'data.train.dataset.select_imgs="{v}" '
                       f"data.val.base_dir='~/data/3d/nerf/nerf_synthetic/{k}' ",
                       shell=True)
        subprocess.run(f"python  nerf_synthetic_render_video.py "
                       f"-f ./data/out/{k}-hf-fail ",
                       shell=True, )
        args.k += args.t
