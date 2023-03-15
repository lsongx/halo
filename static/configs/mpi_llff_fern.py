# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='MPI',
    embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        nb_freqs=10, 
        include_input=True),
    # out_dim = (2*in_dims*nb_freqs + in_dims) if include_input else (2*in_dims*nb_freqs)
    field=dict(
        type='MPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=63, # 2*3*10+3
        use_sin=False),
    render_params=dict())

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='MPILLFFDataset',
            loader=dict(
                type='LLFFLoader',
                colmap_dir='~/data/3d/nerf/nerf_llff_data/fern/',
                im_dir='~/data/3d/nerf/nerf_llff_data/fern/images/',
                bound_factor=0.75,
                center_poses=True,
                spherify_poses=False,
                factor=4,
                ndc=True),
            split='train', 
            size=128,
            # size=-1,
            planes=12,
            holdout=8),
        times=10),
    val=dict(
        type='MPILLFFDataset',
        loader=dict(
            type='LLFFLoader',
            colmap_dir='~/data/3d/nerf/nerf_llff_data/fern/',
            im_dir='~/data/3d/nerf/nerf_llff_data/fern/images/',
            bound_factor=0.75,
            center_poses=True,
            spherify_poses=False,
            factor=4,
            ndc=True),
        split='val', 
        size=-1,
        planes=12,
        holdout=8))

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Exp', gamma=0.1**((1/1000)*(1/100)), by_epoch=False) 
runner = dict(type='EpochBasedRunner', max_epochs=200)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(
    interval=1700, # every 2500 iterations
    render_params=dict())
extra_hooks = [
    dict(type='IterAdjustHook',),
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
