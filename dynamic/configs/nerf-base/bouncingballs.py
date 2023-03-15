_base_ = './default.py'

expname = 'base/dnerf_bouncingballs-400'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='~/data/3d/dnerf/bouncingballs',
    dataset_type='dnerf',
    white_bkgd=True,
)

