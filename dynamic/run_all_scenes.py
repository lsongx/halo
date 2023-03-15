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

 
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='./configs/nerf-small-fewshot/lego.py',)
parser.add_argument("--workspace", type=str, default='./logs/',)
parser.add_argument("--gpus", type=str, required=True,)
args = parser.parse_args()


def run_exp(env, device_id, config, workspace, flags,):
    cmd = (f'python run.py --config {config} --cfg_options ')

    for k, v in flags.items():
        cmd += f"{k}='{v}' "

    print(f'\n\n===> running {cmd}')
    try:
        opt_ret = subprocess.check_output(cmd, shell=True, env=env).decode(sys.stdout.encoding)
        # opt_ret = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, env=env).decode(sys.stdout.encoding)
    except subprocess.CalledProcessError:
        print('Error occurred while running OPT for exp', workspace, 'on', env["CUDA_VISIBLE_DEVICES"])
        return
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    with open(f'{workspace}/{dt_string}.log', 'w') as f:
        f.write(opt_ret)


def process_main(device, queue):
    # Set CUDA_VISIBLE_DEVICES programmatically
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    while True:
        task = queue.get()
        if len(task) == 0:
            break
        run_exp(env, device_id=device, **task)


def determine_frames(basedir):
    with open(os.path.expanduser(os.path.join(basedir, 'transforms_train.json')), 'r') as fp:
        meta = json.load(fp)
    return len(meta['frames'])


if __name__ == '__main__':
    Path(args.workspace).mkdir(parents=True, exist_ok=True)
    pqueue = Queue()
    root = '~/data/3d/dnerf/'
    scenes = ['bouncingballs', 'hellwarrior', 'jumpingjacks', 'hook', 'lego', 
              'mutant', 'standup', 'trex', ]
    param_b = [0]
    task_num = len(scenes)*len(param_b)
    print(f'total task num: {task_num}')
    for i in range(task_num):
        pa = scenes[i % len(scenes)]
        pb = param_b[i // len(scenes)]
        current_task = {
            'config': args.config,
            'workspace': path.join(args.workspace, f'{pa}'),
            'flags': {
                'basedir': args.workspace,
                'expname': f'{pa}',
                'data.datadir': root+pa,
                'model_and_render.lf_reg_path': f'~/code/tineuvox/logs/ori-fewshot-lf/{pa}/fine_last.tar',
            }
        }
        print(current_task)
        pqueue.put(current_task)

    gpus = args.gpus.split(',')
    for _ in gpus:
        pqueue.put({})

    all_procs = []
    for i, gpu in enumerate(gpus):
        process = Process(target=process_main, args=(gpu, pqueue))
        process.daemon = True
        process.start()
        all_procs.append(process)

    for proc in all_procs:
        proc.join()