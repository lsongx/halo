_base_ = './default.py'

expname = 'small/dnerf_hellwarrior-400'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='~/data/3d/dnerf/hellwarrior',
    dataset_type='dnerf',
    white_bkgd=True,
)

