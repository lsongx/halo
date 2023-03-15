_base_ = './default.py'

expname = 'small/dnerf_lego-400'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='~/data/3d/dnerf/lego',
    dataset_type='dnerf',
    white_bkgd=True,
)

