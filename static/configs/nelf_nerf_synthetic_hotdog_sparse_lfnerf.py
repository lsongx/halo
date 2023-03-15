# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='NeLFNeRF360',
    xyz_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        nb_freqs=5, 
        scale=32,
        include_input=False),
    # xyz_embedder=dict(
    #     type='NormalEmbedder',
    #     in_dims=3, 
    #     nb_freqs=5, 
    #     std=8,
    #     scale=5,
    #     include_input=False),
    uv_embedder=dict(
        type='NormalEmbedder',
        in_dims=3,
        nb_freqs=10,
        std=16,
        scale=8,
        include_input=False),
    st_embedder=dict(
        type='NormalEmbedder',
        in_dims=3, 
        nb_freqs=10, 
        std=16, 
        include_input=False),
    dir_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        nb_freqs=4, 
        include_input=False),
    # dir_embedder=dict(
    #     type='NormalEmbedder',
    #     in_dims=3, 
    #     nb_freqs=4, 
    #     std=4,
    #     include_input=False),
    coarse_field=dict(
        type='BaseField',
        nb_layers=8, 
        hid_dims=256, 
        xyz_emb_dims=2*3*5,
        dir_emb_dims=2*3*4,
        use_dirs=False),
    nelf_field=dict(
        type='NeLFEPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*3*(10),
        color_uv_dims=8,
        use_sin=False),
    render_params=dict( # default render cfg; train cfg
        n_samples=128,
        perturb=True,
        alpha_noise_std=1.0,
        inv_depth=False,
        use_dirs=True,
        background='white',
        max_rays_num=1024,),
    rec_loss_weight=1,
    consist_loss_weight=0)

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='AugSyntheticDataset',
            base_dir='~/data/3d/nerf/nerf_synthetic/hotdog', 
            half_res=True,
            batch_size=1024*3,
            background='white',
            select_imgs=['r_16.png', 'r_93.png', 'r_75.png', 'r_86.png', 'r_2.png', 'r_55.png', 'r_73.png', 'r_26.png'],
            precrop_frac=0.5,
            testskip=0,
            split='train'),
        times=100),
    val=dict(
        type='AugSyntheticDataset',
        base_dir='~/data/3d/nerf/nerf_synthetic/hotdog', 
        half_res=True,
        batch_size=-1,
        background='white',
        precrop_frac=0.5,
        testskip=8,
        split='val'))

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# optimizer = dict(type='SGD', lr=1e-1, momentum=0.9)
# optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Poly', power=2, min_lr=1e-4, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=10)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(
    interval=1,
    extra_log='nerf_max_occ_map',
    render_params=dict(
        n_samples=128,
        perturb=False,
        alpha_noise_std=0,
        inv_depth=False,
        use_dirs=True,
        background='white',
        max_rays_num=1024*3,))
extra_hooks = [
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
