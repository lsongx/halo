"""
    Undistort images in Human3.6M and save them alongside (in ".../imageSequence-undistorted/...").

    Usage: `python3 undistort-h36m.py <path/to/Human3.6M-root> <path/to/human36m-multiview-labels.npy> <num-processes>`
"""
# code from https://github.com/karfly/learnable-triangulation-pytorch
import torch
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict

import os, sys


h36m_root = os.path.expanduser('~/data/human36m/training/subject/processed')
number_of_processes = 8
labels_path = os.path.expanduser('~/data/human36m/training/subject/extra/human36m-multiview-labels-GTbboxes.npy')
labels = np.load(labels_path, allow_pickle=True).item()
# print("Dataset length:", len(labels['table']))

n_subjects = len(labels['subject_names'])
n_cameras = len(labels['camera_names'])

# First, prepare: compute distorted meshgrids
print("Computing distorted meshgrids")
meshgrids = np.empty((n_subjects, n_cameras), dtype=object)


class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        self.R = np.array(R).copy()
        assert self.R.shape == (3, 3)

        self.t = np.array(t).copy()
        assert self.t.size == 3
        self.t = self.t.reshape(3, 1)

        self.K = np.array(K).copy()
        assert self.K.shape == (3, 3)

        self.dist = dist
        if self.dist is not None:
            self.dist = np.array(self.dist).copy().flatten()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])


def shot2sample(shot, labels):
    sample = defaultdict(list) # return value

    subject = labels['subject_names'][shot['subject_idx']]
    action = labels['action_names'][shot['action_idx']]
    frame_idx = shot['frame_idx']

    for camera_idx, camera_name in enumerate(labels['camera_names']):
        # load bounding box
        bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1,0,3,2]] # TLBR to LTRB
        bbox_height = bbox[2] - bbox[0]
        if bbox_height == 0:
            # convention: if the bbox is empty, then this view is missing
            continue

        # load image
        image_path = os.path.join(
            os.path.expanduser('~/data/human36m/training/subject/processed'), 
            subject, action, 'imageSequence',
            camera_name, 'img_%06d.jpg' % (frame_idx+1))
        assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
        image = cv2.imread(image_path)

        # load camera
        shot_camera = labels['cameras'][shot['subject_idx'], camera_idx]
        retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)
        sample['cameras'].append(retval_camera)
        sample['images'].append(image)

    # save sample's index
    sample.default_factory = None
    return sample



for sample_idx in tqdm(range(len(labels['table']))):
    subject_idx = labels['table']['subject_idx'][sample_idx]

    if not meshgrids[subject_idx].any():
        bboxes = labels['table']['bbox_by_camera_tlbr'][sample_idx]

        if (bboxes[:, 2] - bboxes[:, 0]).min() > 0: # if == 0, then some camera is missing
            sample = shot2sample(labels['table'][sample_idx], labels)
            assert len(sample['images']) == n_cameras
    
            for camera_idx, (camera, image) in enumerate(zip(sample['cameras'], sample['images'])):
                h, w = image.shape[:2]
                
                fx, fy = camera.K[0, 0], camera.K[1, 1]
                cx, cy = camera.K[0, 2], camera.K[1, 2]
                
                grid_x = (np.arange(w, dtype=np.float32) - cx) / fx
                grid_y = (np.arange(h, dtype=np.float32) - cy) / fy
                meshgrid = np.stack(np.meshgrid(grid_x, grid_y), axis=2).reshape(-1, 2)

                # distort meshgrid points
                k = camera.dist[:3].copy(); k[2] = camera.dist[-1]
                p = camera.dist[2:4].copy()
                
                r2 = meshgrid[:, 0] ** 2 + meshgrid[:, 1] ** 2
                radial = meshgrid * (1 + k[0] * r2 + k[1] * r2**2 + k[2] * r2**3).reshape(-1, 1)
                tangential_1 = p.reshape(1, 2) * np.broadcast_to(meshgrid[:, 0:1] * meshgrid[:, 1:2], (len(meshgrid), 2))
                tangential_2 = p[::-1].reshape(1, 2) * (meshgrid**2 + np.broadcast_to(r2.reshape(-1, 1), (len(meshgrid), 2)))

                meshgrid = radial + tangential_1 + tangential_2

                # move back to screen coordinates
                meshgrid *= np.array([fx, fy]).reshape(1, 2)
                meshgrid += np.array([cx, cy]).reshape(1, 2)

                # cache (save) distortion maps
                meshgrids[subject_idx, camera_idx] = cv2.convertMaps(meshgrid.reshape((h, w, 2)), None, cv2.CV_16SC2)

# Now the main part: undistort images
def undistort_and_save(idx):
    sample = shot2sample(labels['table'][idx], labels)
    
    shot = labels['table'][idx]
    subject_idx = shot['subject_idx']
    action_idx  = shot['action_idx']
    frame_idx   = shot['frame_idx']

    subject = labels['subject_names'][subject_idx]
    action = labels['action_names'][action_idx]

    available_cameras = list(range(len(labels['action_names'])))
    for camera_idx, bbox in enumerate(shot['bbox_by_camera_tlbr']):
        if bbox[2] == bbox[0]: # bbox is empty, which means that this camera is missing
            available_cameras.remove(camera_idx)

    for camera_idx, image in zip(available_cameras, sample['images']):
        camera_name = labels['camera_names'][camera_idx]

        output_image_folder = os.path.join(
            h36m_root, subject, action, 'imageSequence-undistorted', camera_name)
        output_image_path = os.path.join(output_image_folder, 'img_%06d.jpg' % (frame_idx+1))
        os.makedirs(output_image_folder, exist_ok=True)

        meshgrid_int16 = meshgrids[subject_idx, camera_idx]
        image_undistorted = cv2.remap(image, *meshgrid_int16, cv2.INTER_CUBIC)

        cv2.imwrite(output_image_path, image_undistorted)

print(f"Undistorting images using {number_of_processes} parallel processes")
cv2.setNumThreads(1)
import multiprocessing

pool = multiprocessing.Pool(number_of_processes)
for _ in tqdm(pool.imap_unordered(
    undistort_and_save, range(len(labels['table'])), chunksize=10), total=len(labels['table'])):
    pass

pool.close()
pool.join()
