from math import ceil
from multiprocessing import Process, Queue
import sys
import os
from os import path, listdir
import argparse
import json
import subprocess
import random
import numpy as np
from pathlib import Path
from datetime import datetime


# root = os.path.expanduser('~/code/tineuvox/logs/ori-fewshot')
root = os.path.expanduser('~/code/tineuvox/logs/ori-fewshot-lfhf')
folders = list(os.listdir(root))

all_p = {}
all_s = {}
all_v = {}
all_a = {}
for f in folders:
    for k in os.listdir(f'{root}/{f}'):
        if '.log' in k:
            break
    with open(f'{root}/{f}/{k}', 'r') as log:
        text = log.read().split('\n')
    psnr = text[-9].split('psnr ')[-1].split(' (a')[0]
    ssim = text[-8].split('ssim ')[-1].split(' (a')[0]
    vgg = text[-7].split('vgg) ')[-1].split(' (a')[0]
    alex = text[-6].split('alex) ')[-1].split(' (a')[0]
    all_p[f] = float(psnr)
    all_s[f] = float(ssim)
    all_v[f] = float(vgg)
    all_a[f] = float(alex)

names = [k for k in all_p.keys()]
for k in range(len(names)):
    names[k] = names[k].capitalize() 
print(' & '.join(names)+' & Avg')
print(' & '.join([f'{k:.3f}' for k in all_p.values()])+f' & {np.asarray(list(all_p.values())).mean():.3f}')
print(' & '.join([f'{k:.3f}' for k in all_s.values()])+f' & {np.asarray(list(all_s.values())).mean():.3f}')
print(' & '.join([f'{k:.3f}' for k in all_v.values()])+f' & {np.asarray(list(all_v.values())).mean():.3f}')
print(' & '.join([f'{k:.3f}' for k in all_a.values()])+f' & {np.asarray(list(all_a.values())).mean():.3f}')

