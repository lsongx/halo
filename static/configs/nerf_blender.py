# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='NeRF',
    xyz_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        nb_freqs=10, 
        include_input=True),
    dir_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        nb_freqs=4, 
        include_input=True),
    # out_dim = (2*in_dims*nb_freqs + in_dims) if include_input else (2*in_dims*nb_freqs)
    coarse_field=dict(
        type='BaseField',
        nb_layers=8, 
        hid_dims=256, 
        xyz_emb_dims=63, # 2*3*10+3
        dir_emb_dims=27, # 2*3*4+3
        use_dirs=True),
    fine_field=dict(
        type='BaseField',
        nb_layers=8, 
        hid_dims=256, 
        xyz_emb_dims=63, # 2*3*10+3
        dir_emb_dims=27, # 2*3*4+3
        use_dirs=True),
    render_params=dict( # default render cfg; train cfg
        n_samples=64,
        n_importance=128,
        perturb=True,
        alpha_noise_std=1.0,
        inv_depth=False,
        white_bG=True,
        use_dirs=True,
        max_rays_num=1024*3,
        ndc=False))

# dataset settings
data = dict(
    samples_per_gpu=1024,
    workers_per_gpu=2,
    train=dict(
        type='NeRFDataset',
        loader=dict(
            type='BlenderLoader',
            root_dir='../blender-data/blender-output/sit_lauGh_static_19cam_360x480_24x32_focal50.0_64spl_step1/',
            bound_factor=0.75,
            center_poses=True,
            spherify_poses=False,
            ndc=False),
        split='train', 
        holdout=8),
    val=dict(
        type='NeRFDataset',
        loader=dict(
            type='BlenderLoader',
            root_dir='../blender-data/blender-output/sit_lauGh_static_19cam_360x480_24x32_focal50.0_64spl_step1/',
            bound_factor=0.75,
            center_poses=True,
            spherify_poses=False,
            ndc=False),
        split='val', 
        holdout=8))

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Exp', gamma=0.1**((1/1000)*(1/250)), by_epoch=False) 
runner = dict(type='EpochBasedRunner', max_epochs=20)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
        dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(
    interval=1,
    render_params=dict(
        n_samples=64,
        n_importance=128,
        perturb=False,
        alpha_noise_std=0,
        inv_depth=False,
        white_bG=True,
        use_dirs=True,
        max_rays_num=1024*2,
        ndc=False))
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
