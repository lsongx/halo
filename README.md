# halo
This repo is for replicating the results reported in our paper "Harnessing Low-Frequency Neural Fields for Few-Shot View Synthesis".

## training
### static scenes
All scripts for replicating the numbers reported in the paper can be found at `static/scripts`.

### dynamic scenes
Run with `run_all_scenes.py` for experiments on all scenes in D-NeRF. All config files can be found at
```
dynamic
    + configs
        + nerf-small-fewshot       # fewshot baseline
        + nerf-small-fewshot-lf    # fewshot with low-freq only
        + nerf-small-fewshot-lfhf  # fewshot with high-freq added
```
