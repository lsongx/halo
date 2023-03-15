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
    'oneside-lego': "['r_58.png','r_5.png','r_2.png','r_8.png','r_9.png','r_10.png','r_16.png','r_34.png','r_35.png','r_40.png','r_52.png','r_53.png','r_54.png','r_60.png',]",
}

if __name__ == '__main__':
    while args.k < len(scene):
        k = list(scene.keys())[args.k]
        v = scene[k]
        subprocess.run(f"python -m torch.distributed.launch --nproc_per_node=1 train.py "
                       f"--seed 1 --use_fp16 1 --config ./configs/nelf_nerf_synthetic_lego_sparse_hf_fail.py "
                       f"--work_dir ./data/out/{k}-hf-fail --port {25666+args.k} "
                       f"--cfg-options data.train.dataset.base_dir='~/data/3d/nerf/nerf_synthetic/lego' "
                       f'data.train.dataset.select_imgs="{v}" '
                       f"data.val.base_dir='~/data/3d/nerf/nerf_synthetic/lego' ",
                       shell=True)
        subprocess.run(f"python  nerf_synthetic_render_video.py "
                       f"-f ./data/out/{k}-hf-fail ",
                       shell=True, )
        args.k += args.t
