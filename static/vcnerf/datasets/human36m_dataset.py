import os
import numpy as np
import torch
from torch.utils.data import Dataset

from mmcv.utils import build_from_cfg
from .builder import DATASETS, LOADERS


@DATASETS.register_module()
class Human36MDataset(Dataset):
    EXCLUDE_ACTIONS = ['eating', 'posing', 'phoning', 'sitting', 'smoking', ]

    def __init__(self, 
                 label_path, 
                 image_root, 
                 loader, 
                 split, 
                 holdout=0.8,
                 frame_freq=30,
                 subject_idx=0, 
                 rays_per_img=1024,
                 infer_bbox_from_color=True,
                 in_bbox_ratio=0.7):
        """
        image_root: root for processed images; 
        label_path: after loading (label dict)
            dict_keys(['subject_names', 'camera_names', 'action_names', 'cameras', 'table'])
            subject_names: ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
            camera_names: ['54138969', '55011271', '58860488', '60457274']
            action_names: ['Sitting-1', ...]
            cameras: (7, 4)[subject, camera]; each entry: ['R', 't', 'K', 'dist']
            table: (161362,); each entry: ['subject_idx', 'action_idx', 'frame_idx', 
                                           'keypoints', 'keypoints_full', 'bbox_by_camera_tlbr']
        holdout: percentage for val
        """
        super().__init__()
        assert split in ('train', 'val', 'test'), split
        self.label = np.load(os.path.expanduser(label_path), allow_pickle=True).item()
        import ipdb; ipdb.set_trace()
        self.image_root = os.path.expanduser(image_root)
        self.loader = build_from_cfg(loader, LOADERS)
        self.w = self.loader.w
        self.h = self.loader.h
        self.split = split
        self.holdout = holdout
        self.rays_per_img = rays_per_img
        self.infer_bbox_from_color = infer_bbox_from_color
        self.in_bbox_ratio = in_bbox_ratio

        subject_mask = self.label['table']['subject_idx']==subject_idx
        exclude_mask = subject_mask
        for i, v in enumerate(self.label['action_names']):
            for e in self.EXCLUDE_ACTIONS:
                if e.lower() in v.lower():
                    exclude_mask = exclude_mask & (self.label['table']['action_idx']!=i)
        ori_samples_idx = np.nonzero(exclude_mask)[0]
        samples_idx = np.asarray([v for i, v in enumerate(ori_samples_idx) if i%frame_freq==0])
        split_idx = int(self.holdout*len(samples_idx))

        if self.split == 'train':
            self.samples_idx = samples_idx[:split_idx]
        else:
            self.samples_idx = samples_idx[split_idx:]

    def __len__(self):
        return len(self.samples_idx)*len(self.label['camera_names'])

    def __getitem__(self, idx):
        sample_idx = self.samples_idx[idx // 4]
        camera_idx = idx % 4
        shot = self.label['table'][sample_idx]
        subject = self.label['subject_names'][shot['subject_idx']]
        action = self.label['action_names'][shot['action_idx']]
        frame_idx = shot['frame_idx']+1

        camera = self.label['camera_names'][camera_idx]
        image_name = f'{subject}-{action}-{camera}-frame{frame_idx:06d}.jpg'
        image_path = os.path.join(self.image_root, image_name)

        shot_camera = self.label['cameras'][shot['subject_idx'], camera_idx]
        intrinsics = shot_camera['K']
        extrinsics = np.hstack([shot_camera['R'], shot_camera['t']])

        if self.split == 'train':
            rays_ori, rays_dir, rays_color = \
                self.loader.load_img_sample_rays(image_path, intrinsics, extrinsics,
                                                 shot['bbox_by_camera_tlbr'][camera_idx],
                                                 self.infer_bbox_from_color,
                                                 self.rays_per_img, self.in_bbox_ratio)
        else:
            rays_ori, rays_dir, rays_color = \
                self.loader.load_img_all_rays(image_path, intrinsics, extrinsics)

        return {'rays_ori': rays_ori, 
                'rays_dir': rays_dir, 
                'rays_color': rays_color,
                'pose': shot['keypoints_full']}

