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
    render_cfg=dict( # default render cfg; train cfg
        n_samples=64,
        n_importance=128,
        perturb=True,
        alpha_noise_std=1.0,
        inv_depth=False,
        use_dirs=True,
        max_rays_num=1024*3,))

# dataset settings
data = dict(
    samples_per_gpu=1024,
    workers_per_gpu=2,
    train=dict(
        type='Human36MDataset',
        label_path='~/data/human36m/training/subject/extra/human36m-multiview-labels-GTbboxes.npy', 
        image_root='~/data/human36m/training/subject/masked', 
        loader=dict(type='Human36MLoader', h=512, w=512),
        split='train',
        holdout=0.8,
        frame_freq=30,
        subject_idx=0, 
        rays_per_img=1024,
        infer_bbox_from_color=True,
        in_bbox_ratio=0.7),
    val=dict(
        type='Human36MDataset',
        label_path='~/data/human36m/training/subject/extra/human36m-multiview-labels-GTbboxes.npy', 
        image_root='~/data/human36m/training/subject/masked', 
        loader=dict(type='Human36MLoader', h=512, w=512),
        split='val',
        holdout=0.8,
        frame_freq=30,
        subject_idx=0, 
        rays_per_img=1024,
        infer_bbox_from_color=True,
        in_bbox_ratio=0.7))

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
    interval=2500,
    render_cfg=dict(
        n_samples=64,
        n_importance=128,
        perturb=False,
        alpha_noise_std=0,
        inv_depth=False,
        use_dirs=True,
        max_rays_num=1024*2,))
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
