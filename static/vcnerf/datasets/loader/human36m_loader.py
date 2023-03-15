import os
import random
import cv2
import numpy as np

from vcnerf.utils import get_root_logger
from ..utils import collect_rays, center_poses
from ..builder import LOADERS


def update_after_resize(image_shape, new_image_shape, K):
    height, width = image_shape
    new_height, new_width = new_image_shape

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    new_fx = fx * (new_width / width)
    new_fy = fy * (new_height / height)
    new_cx = cx * (new_width / width)
    new_cy = cy * (new_height / height)

    new_K = K.copy()
    new_K[0, 0], new_K[1, 1], new_K[0, 2], new_K[1, 2] = new_fx, new_fy, new_cx, new_cy
    return new_K


def update_bbox(image_shape, new_image_shape, bbox_tlbr):
    height, width = image_shape
    new_height, new_width = new_image_shape
    bbox_tlbr[[0,2]] *= new_height / height
    bbox_tlbr[[1,3]] *= new_width / width
    return bbox_tlbr


def infer_bbox(image, background_larger_thresh=240):
    mask_all = ~(image > background_larger_thresh)
    mask = mask_all[...,0] & mask_all[...,1] & mask_all[...,0]
    all_index = np.nonzero(mask)
    t = all_index[0].min()
    d = all_index[0].max()
    l = all_index[1].min()
    r = all_index[1].max()
    return np.asarray([t,l,d,r], dtype=np.float)


@LOADERS.register_module()
class Human36MLoader(object):

    def __init__(self, h, w):
        super().__init__()
        self.logger = get_root_logger()
        self.h = h
        self.w = w

    def generate_sample_mask(self, bbox_tlbr, num_rays, in_bbox_ratio):
        bbox_tlbr = bbox_tlbr.astype(np.int)
        num_all_rays = self.h*self.w
        full_mask = np.zeros([self.h, self.w], dtype=bool)
        out_bbox_mask = ~full_mask
        out_bbox_mask[bbox_tlbr[0]:bbox_tlbr[2], bbox_tlbr[1]:bbox_tlbr[3]] = 0
        in_bbox_mask = ~out_bbox_mask

        bbox_rays = (bbox_tlbr[2]-bbox_tlbr[0])*(bbox_tlbr[3]-bbox_tlbr[1])
        bbox_samples = int(num_rays * in_bbox_ratio)
        in_bbox_idx = np.nonzero(in_bbox_mask)
        sample_idx = np.random.choice(range(in_bbox_idx[0].shape[0]), bbox_samples, replace=False)
        in_bbox_idx_sampled = (in_bbox_idx[0][sample_idx], in_bbox_idx[1][sample_idx])
        full_mask[in_bbox_idx_sampled[0], in_bbox_idx_sampled[1]] = 1

        out_samples = num_rays - bbox_samples
        out_bbox_idx = np.nonzero(out_bbox_mask)
        sample_idx = np.random.choice(range(out_bbox_idx[0].shape[0]), out_samples, replace=False)
        out_bbox_idx_sampled = (out_bbox_idx[0][sample_idx], out_bbox_idx[1][sample_idx])
        full_mask[out_bbox_idx_sampled[0], out_bbox_idx_sampled[1]] = 1

        assert full_mask.sum() == num_rays, 'Error when generating mask'
        return full_mask

    def load_img_sample_rays(self, 
                             image_path, 
                             intrinsics, 
                             extrinsics, 
                             bbox_tlbr, 
                             infer_bbox_from_color,
                             num_rays,
                             in_bbox_ratio=0.7):
        """
        intrinsics: [3, 3] array
        extrinsics: [3, 4] array, world2camera
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image = cv2.resize(image, (self.w, self.h))

        intrinsics = update_after_resize(ori_size, (self.h, self.w), intrinsics)
        if infer_bbox_from_color:
            bbox_tlbr = infer_bbox(image, background_larger_thresh=240)
        else:
            bbox_tlbr = update_bbox(ori_size, (self.h, self.w), bbox_tlbr)
        m = self.generate_sample_mask(bbox_tlbr, num_rays, in_bbox_ratio)

        m = m.reshape([-1])
        rays_color = image.reshape([-1, 3])
        rays_ori, rays_dir = self.collect_rays(intrinsics, extrinsics)
        return rays_ori[m,:], rays_dir[m,:], rays_color[m,:]

    def load_img_all_rays(self, image_path, intrinsics, extrinsics):
        """
        intrinsics: [3, 3] array
        extrinsics: [3, 4] array, world2camera
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image = cv2.resize(image, (self.w, self.h))

        intrinsics = update_after_resize(ori_size, (self.h, self.w), intrinsics)
        rays_color = image.reshape([-1, 3])
        rays_ori, rays_dir = self.collect_rays(intrinsics, extrinsics)
        
        return rays_ori, rays_dir, rays_color

    def collect_rays(self, intrinsics, extrinsics):
        """
        intrinsics: [3, 3] array
        extrinsics: [3, 4] array, world2camera
        """
        h, w = self.h, self.w # make things shorter
        meshgrid = np.meshgrid(range(w), range(h), indexing='xy')
        ones = np.ones(w*h) # w*h
        pix_coords = np.stack([meshgrid[0].reshape(-1), meshgrid[1].reshape(-1), ones], axis=0) # [3, w*h]

        inv_K = np.linalg.inv(intrinsics)
        # cam points with depth=1
        cam_points = inv_K @ pix_coords.astype(np.float32)
        # homo_cam_points = np.concatenate((cam_points, ones.copy()), axis=0)

        dirs = cam_points
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        extrinsics = np.concatenate([extrinsics, bottom], 0)
        c2w = np.linalg.inv(extrinsics)
        # ray directions in world coordinate system
        # rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1) # maybe slower
        rays_dir = c2w[:3, :3] @ dirs # [3, w*h]
        rays_dir = rays_dir.transpose([1,0]) # [w*h, 3]
        # ray origins in world coordinate system
        rays_ori = np.broadcast_to(c2w[:3, -1], np.shape(rays_dir))

        return rays_ori, rays_dir

