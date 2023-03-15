import subprocess
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


select_imgs = "['r_2.png', 'r_16.png', 'r_93.png', 'r_55.png', 'r_73.png', 'r_86.png', 'r_26.png', 'r_75.png']"
input_views = {
    '2': "['r_2.png', 'r_16.png']",
    '3': "['r_2.png', 'r_16.png', 'r_93.png']",
    '4': "['r_2.png', 'r_16.png', 'r_93.png', 'r_55.png',]",
    '5': "['r_2.png', 'r_16.png', 'r_93.png', 'r_55.png', 'r_73.png',]",
    '6': "['r_2.png', 'r_16.png', 'r_93.png', 'r_55.png', 'r_73.png', 'r_86.png',]",
    '7': "['r_2.png', 'r_16.png', 'r_93.png', 'r_55.png', 'r_73.png', 'r_86.png', 'r_26.png']",
}

while args.k < len(input_views):
    freq = list(input_views.keys())[args.k]
    subprocess.run(f"python -m torch.distributed.launch --nproc_per_node=1 train.py "
                   f"--seed 1 --use_fp16 1 --config ./configs/nelf_nerf_synthetic_hotdog_sparse_lfnerf.py "
                   f"--work_dir ./data/out/views-lego-{freq} --port {25666+args.k} "
                   f"--cfg-options data.train.dataset.base_dir='~/data/3d/nerf/nerf_synthetic/lego' "
                   f'data.train.dataset.select_imgs="{input_views[freq]}" '
                   f"data.val.base_dir='~/data/3d/nerf/nerf_synthetic/lego' ",
                   shell=True)
    args.k += args.t
