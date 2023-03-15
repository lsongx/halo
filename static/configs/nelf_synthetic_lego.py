
# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='NeLF',
    embedder=dict(
        type='BaseEmbedder',
        in_dims=4, 
        nb_freqs=32, 
        include_input=True),
    # out_dim = (2*in_dims*nb_freqs + in_dims) if include_input else (2*in_dims*nb_freqs)
    field=dict(
        type='NeLFField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=260, # 2*4*32+4
        use_sin=False),
    render_params=dict())

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='SyntheticSphereCoordDataset',
            base_dir='~/data/3d/nerf/nerf_synthetic/lego', 
            half_res=True,
            batch_size=-1,
            background='white',
            precrop_frac=0.5,
            testskip=8,
            split='train'),
        times=100),
    val=dict(
        type='SyntheticSphereCoordDataset',
        base_dir='~/data/3d/nerf/nerf_synthetic/lego', 
        half_res=True,
        batch_size=-1,
        background='white',
        precrop_frac=0.5,
        testskip=8,
        split='val'))

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Exp', gamma=0.1**(1/75000), by_epoch=False) 
runner = dict(type='EpochBasedRunner', max_epochs=200)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=300,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(
    interval=5000,
    render_params=dict())
extra_hooks = [
    dict(type='IterAdjustHook',),
    dict(
        type='DatasetParamAdjustHook',
        param_name_adjust_iter_value = [
            # ('precrop_frac', 0, 0.9),
            ('precrop_frac', 500, 1),],)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
