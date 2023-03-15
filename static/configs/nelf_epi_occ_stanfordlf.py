# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='NeLFEPIOcc',
    embedder=dict(
        type='NormalEmbedder',
        in_dims=2, 
        nb_freqs=128, 
        std=32,
        # lf_nb_freqs=32,
        # lf_std=16,
        # max_scale_iter=2200*100,
        # include_input=True),
        include_input=False),
    # embedder=dict(
    #     type='BaseEmbedder',
    #     in_dims=2, 
    #     nb_freqs=10, 
    #     include_input=False),
    epi_embedder=dict(
        type='NormalEmbedder',
        in_dims=1, 
        nb_freqs=10,
        std=16,
        include_input=False),
    # hf_color_embedder=dict(
    #     type='NormalEmbedder',
    #     in_dims=8, 
    #     nb_freqs=10, 
    #     std=1024,
    #     include_input=False),
    hf_color_embedder=None,
    # out_dim = (2*in_dims*nb_freqs + in_dims) if include_input else (2*in_dims*nb_freqs)
    field=dict(
        type='NeLFEPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(128),
        color_dims=8,
        color_uv_dims=8,
        positive_color_code=True,
        # hf_color_subnet_dim=2*8*10,
        hf_color_subnet_dim=0,
        use_sin=True),
    occ_field=dict(
        type='NeLFEPIOccField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(128),
        epi_emb_dims=2*10, 
        color_dims=8,
        positive_color_code=True,
        use_sin=True),
    epi_reg_weight=2,
    epi_sample_num=128,
    occ_consist_weight=0,
    # pixel_move=40,
    # pixel_move=20,
    pixel_move=15,
    near=0.1343, # 37(delta_uv[max]) 5(delta_st=-delta_img(-30)+delta_uv[min]); atan(5/37)
    far=1.0894, # 35(delta_uv[min]) 67(delta_st=delta_img(30)+delta_uv[max]); atan(67/35)
    epi_converge_iter=2200*100, 
    epi_converge_range=0.2,
    # epi_converge_range=1,
    render_params=dict(aug_points=512, perturb=True))

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='BatchStanfordLFDataset',
            # base_dir='~/data/3d/StandfordLF/cards-fine',
            # base_dir='~/data/3d/StandfordLF/cards-coarse', 
            # base_dir='~/data/3d/StandfordLF/lego-truck',
            base_dir='~/data/3d/StandfordLF/lego-knights', 
            # base_dir='~/data/3d/StandfordLF/eucalyptus-flowers', 
            # downsample=1,
            downsample=2,
            # downsample=4,
            scale=1024,
            # batch_size=int(1024*1.8),
            batch_size=int(1024),
            # batch_size=512,
            testskip=8,
            perturb=True,
            split='train',),
        times=200),
        # times=100),
        # times=1),
    val=dict(
        type='BatchStanfordLFDataset',
        # base_dir='~/data/3d/StandfordLF/cards-fine',
        # base_dir='~/data/3d/StandfordLF/cards-coarse', 
        # base_dir='~/data/3d/StandfordLF/lego-truck',
        base_dir='~/data/3d/StandfordLF/lego-knights', 
        # base_dir='~/data/3d/StandfordLF/eucalyptus-flowers', 
        # downsample=1,
        downsample=2,
        # downsample=4,
        scale=1024,
        batch_size=-1,
        testskip=8,
        split='val',))

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
# optimizer = dict(
#     # type='AdamW', lr=5e-4, 
#     type='Adam', lr=5e-4, betas=(0.9, 0.999),
#     paramwise_cfg=dict(custom_keys={'.field.color_code_hf_subnet': dict(lr_mult=0.1)}))
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='Exp', gamma=0.1**(1/75000), by_epoch=False)
# lr_config = dict(policy='Poly', min_lr=5e-6, by_epoch=False)
lr_config = dict(policy='Step', step=[50,100,150], gamma=0.5, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
# lr_config = dict(policy='Step', step=[150,200,250], gamma=0.5, by_epoch=True)
# runner = dict(type='EpochBasedRunner', max_epochs=300)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=30)
log_config = dict(
    interval=50,
    # interval=1,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(
    interval=1,
    extra_log='epi_map',
    render_params=dict(batch_ray_forward=1024, perturb=False))
extra_hooks = [
    dict(type='IterAdjustHook',), 
    dict(type='EPIUpdateModelByDataset')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
