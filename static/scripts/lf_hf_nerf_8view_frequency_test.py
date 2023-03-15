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
freq_tests = {
    's1n5': 'model.xyz_embedder.scale=1 ',
    's4n5': 'model.xyz_embedder.scale=4 ',
    's8n5': 'model.xyz_embedder.scale=8 ',
    's16n5': 'model.xyz_embedder.scale=16 ',
    's64n5': 'model.xyz_embedder.scale=64 ',
    's128n5': 'model.xyz_embedder.scale=128 ',
    's8n10': 'model.xyz_embedder.scale=8 model.xyz_embedder.nb_freqs=10 model.coarse_field.xyz_emb_dims=60 ',
    's16n10': 'model.xyz_embedder.scale=16 model.xyz_embedder.nb_freqs=10 model.coarse_field.xyz_emb_dims=60 ',
    's64n10': 'model.xyz_embedder.scale=64 model.xyz_embedder.nb_freqs=10 model.coarse_field.xyz_emb_dims=60 ',
    's128n10': 'model.xyz_embedder.scale=128  model.xyz_embedder.nb_freqs=10 model.coarse_field.xyz_emb_dims=60 ',
}

while args.k < len(freq_tests):
    freq = list(freq_tests.keys())[args.k]
    subprocess.run(f"python -m torch.distributed.launch --nproc_per_node=1 train.py "
                   f"--seed 1 --use_fp16 1 --config ./configs/nelf_nerf_synthetic_hotdog_sparse_lfnerf.py "
                   f"--work_dir ./data/out/freqtest-lego-{freq} --port {25666+args.k} "
                   f"--cfg-options data.train.dataset.base_dir='~/data/3d/nerf/nerf_synthetic/lego' "
                   f'data.train.dataset.select_imgs="{select_imgs}" '
                   f"data.val.base_dir='~/data/3d/nerf/nerf_synthetic/lego' "+freq_tests[freq],
                   shell=True)
    args.k += args.t
