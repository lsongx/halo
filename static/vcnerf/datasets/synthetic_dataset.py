from inspect import FullArgSpec
import math
import os
import os.path as osp
import random
from scipy import spatial
import json
import numpy as np
import imageio
import cv2
import torch

from vcnerf.utils import get_root_logger
from .builder import DATASETS


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1).to(c2w.device)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_dir = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_ori = np.broadcast_to(c2w[:3,-1], np.shape(rays_dir))
    return rays_ori, rays_dir

def sphere2rot(coord):
    x = np.array([-coord[1], coord[0], 0])
    # a = coord + x/np.linalg.norm(x)
    a = x/np.linalg.norm(x)
    r = np.linalg.norm(coord)
    new_z = r/(coord[2]/r)
    b = np.array([0,0,new_z])-coord
    b = b/np.linalg.norm(b)
    c = coord/np.linalg.norm(coord)
    rotation_matrix = np.stack([a,b,c], 0)
    return np.linalg.inv(rotation_matrix)

@DATASETS.register_module()
class SyntheticDataset(object):
    def __init__(self, 
                 base_dir, 
                 split, 
                 half_res,
                 batch_size,
                 batching=False,
                 select_imgs=None,
                 background='white',
                 precrop_frac=0.5,
                 num_render_img=40,
                 to_cuda=False,
                 testskip=8,):
        super().__init__()
        self.logger = get_root_logger()
        self.base_dir = os.path.expanduser(base_dir)
        self.split = split
        self.half_res = half_res
        self.batch_size = batch_size
        self.select_imgs = select_imgs
        self.background = background
        self.precrop_frac = precrop_frac
        self.to_cuda = to_cuda
        self.testskip = testskip
        self.batching = batching
        self.num_render_img = num_render_img

        self.__init_dataset()

    def __init_dataset(self):
        file = osp.join(self.base_dir, f'transforms_{self.split}.json')
        with open(file, 'r') as fp:
            meta = json.load(fp)

        imgs = []
        poses = []
        if self.split=='train' or self.testskip==0:
            skip = 1
        else:
            skip = self.testskip

        all_names = []
        for frame in meta['frames'][::skip]:
            fname = osp.join(self.base_dir, frame['file_path'] + '.png')
            if self.select_imgs is not None:
                for i in self.select_imgs:
                    if i in fname:
                        imgs.append(imageio.imread(fname))
                        poses.append(np.array(frame['transform_matrix']))
                        all_names.append(frame['file_path'])
            else:
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
                all_names.append(frame['file_path'])
        self.imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        self.poses = np.array(poses).astype(np.float32)
        self.rad = (self.poses[:,:3,3]**2).sum(1).mean()
        self.logger.info(f'loaded images: {all_names} with '
                         f'rad mean {(self.poses[:,:3,3]**2).sum(1).mean()} '
                         f'rad std {(self.poses[:,:3,3]**2).sum(1).std()}')

        self.h, self.w = self.imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
        self.render_poses = torch.stack(
            [pose_spherical(angle, -30.0, 4.0) 
            for angle in np.linspace(-180,180,self.num_render_img+1)[:-1]], 0)
    
        self.h = int(self.h)
        self.w = int(self.w)

        self.near = 2.
        self.far = 6.

        if self.half_res:
            self.h = self.h//2
            self.w = self.w//2
            self.focal = self.focal/2.

            imgs_half_res = np.zeros((self.imgs.shape[0], self.h, self.w, 4))
            for i, img in enumerate(self.imgs):
                # imgs_half_res[i] = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
                imgs_half_res[i] = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            self.imgs = imgs_half_res

        if self.background == 'white':
            self.imgs = self.imgs[...,:3]*self.imgs[...,-1:] + (1.-self.imgs[...,-1:])
        elif self.background == 'black':
            self.imgs = self.imgs[...,:3]*self.imgs[...,-1:]
        else:
            self.imgs = self.imgs[...,:3]

        self.imgs = torch.tensor(self.imgs).float()
        self.poses = torch.tensor(self.poses).float()
        if self.to_cuda:
            self.imgs.cuda()
            self.poses.cuda()
        self.length = len(self.imgs)
        if self.batching:
            all_rays_ori = []
            all_rays_dir = []
            for pose in self.poses[:, :3, :4]:
                ro, rd = get_rays(self.h, self.w, self.focal, pose)
                all_rays_ori.append(ro)
                all_rays_dir.append(rd)
            perm = torch.randperm(self.length*self.h*self.w)
            self.all_rays_ori = torch.stack(all_rays_ori,0).view([-1,3])[perm]
            self.all_rays_dir = torch.stack(all_rays_dir,0).view([-1,3])[perm]
            self.all_rays_color = self.imgs.view([-1,3])[perm]
            self.length = math.ceil(len(perm)/self.batch_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.batching:
            e = min(idx*self.batch_size+self.batch_size, 
                    self.all_rays_color.shape[0])
            s = e-self.batch_size
            return {'rays_ori': self.all_rays_ori[s:e],
                    'rays_dir': self.all_rays_dir[s:e],
                    'rays_color': self.all_rays_color[s:e],
                    'near': self.near, 'far': self.far}
        if idx >= len(self.imgs):
            target = self.imgs[0]
        else:
            target = self.imgs[idx]
        pose = self.poses[idx, :3, :4]
        rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose)

        if self.batch_size == -1:
            rays_color = target.view([-1,3])  # (N, 3)
            return {'rays_ori': rays_ori.view([-1,3]), 
                    'rays_dir': rays_dir.view([-1,3]), 
                    'rays_color': rays_color,
                    'near': self.near, 'far': self.far}

        # if self.precrop_frac < 1:
        #     dH = int(self.h//2 * self.precrop_frac)
        #     dW = int(self.w//2 * self.precrop_frac)
        #     coords = torch.stack(
        #         torch.meshgrid(
        #             torch.linspace(self.h//2 - dH, self.h//2 + dH - 1, 2*dH), 
        #             torch.linspace(self.w//2 - dW, self.w//2 + dW - 1, 2*dW)
        #         ), -1)
        # else:
        #     coords = torch.stack(torch.meshgrid(
        #         torch.linspace(0, self.h-1, self.h), 
        #         torch.linspace(0, self.w-1, self.w)), -1)
        # coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        # select_inds = np.random.choice(coords.shape[0], size=[self.batch_size], replace=False)  # (N,)
        # select_coords = coords[select_inds].long()  # (N, 2)
        # rays_ori = rays_ori[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)
        # rays_dir = rays_dir[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)
        # rays_color = target[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)
        if self.precrop_frac < 1:
            dH = int(self.h//2 * self.precrop_frac)
            dW = int(self.w//2 * self.precrop_frac)
            rays_ori = rays_ori[self.h//2-dH:self.h//2+dH, self.w//2-dW:self.w//2+dW]
            rays_dir = rays_dir[self.h//2-dH:self.h//2+dH, self.w//2-dW:self.w//2+dW]
            target = target[self.h//2-dH:self.h//2+dH, self.w//2-dW:self.w//2+dW]
            n = dH*dW*4
        else:
            n = self.h*self.w
        select_mask = torch.randperm(n)[:self.batch_size]
        rays_ori = rays_ori.reshape([-1,3])[select_mask]
        rays_dir = rays_dir.reshape([-1,3])[select_mask]
        rays_color = target.reshape([-1,3])[select_mask]

        return {'rays_ori': rays_ori, 'rays_dir': rays_dir, 
                'rays_color': rays_color, 
                'near': self.near, 'far': self.far}


@DATASETS.register_module()
class SyntheticWithPoseDataset(SyntheticDataset):

    def __getitem__(self, idx):
        target = self.imgs[idx]
        pose = self.poses[idx, :3, :4]
        rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose)

        if self.batch_size == -1:
            rays_color = target.view([-1,3])  # (N, 3)
            select_mask = torch.ones([self.h, self.w]).bool()
            return {'rays_ori': rays_ori.view([-1,3]), 
                    'rays_dir': rays_dir.view([-1,3]), 
                    'rays_color': rays_color, 
                    'ndc_rays_ori': 0, 
                    'ndc_rays_dir': 0,
                    'h': self.h, 'w': self.w,
                    'pose': pose,
                    'select_mask': select_mask,
                    'near': self.near, 
                    'far': self.far}

        select_mask = torch.rand([self.h, self.w])
        if self.precrop_frac < 1:
            dH = int(self.h//2 * self.precrop_frac)
            dW = int(self.w//2 * self.precrop_frac)
            select_mask[dH:-dH, dW:-dW] *= -1

        percent_value = select_mask.view([-1]).kthvalue(self.batch_size).values
        select_mask = select_mask <= percent_value
        rays_ori = rays_ori[select_mask]  # (N, 3)
        rays_dir = rays_dir[select_mask]  # (N, 3)
        rays_color = target[select_mask]  # (N, 3)

        return {'rays_ori': rays_ori, 
                'rays_dir': rays_dir, 
                'rays_color': rays_color, 
                'ndc_rays_ori': 0, 
                'ndc_rays_dir': 0,
                'h': self.h, 'w': self.w,
                'pose': pose,
                'select_mask': select_mask,
                'near': self.near, 
                'far': self.far}


@DATASETS.register_module()
class AugSyntheticDataset(SyntheticDataset):
    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        # pose = self.poses[idx, :3, :4]
        # aug_axis_degree = (random.random()-0.5)*7+3
        # aug_mat = spatial.transform.Rotation.from_euler('z', aug_axis_degree, degrees=True)
        # aug_mat = torch.tensor(aug_mat.as_matrix()).float()
        # aug_pose = aug_mat @ pose
        # aug_rays_ori, aug_rays_dir = get_rays(self.h, self.w, self.focal, aug_pose)
        # aug_rays_ori = sample_nearby_point_on_sphere(aug_rays_ori, np.pi/6)

        # pose0 = int(random.random()*len(self.poses))
        # pose1 = int(random.random()*len(self.poses))
        # aug_pose  = (self.poses[pose0, :3, :4]+self.poses[pose1, :3, :4])/2
        # aug_rays_ori, aug_rays_dir = get_rays(self.h, self.w, self.focal, aug_pose)

        rand_point = np.array([random.random()-0.5, random.random()-0.5, abs(random.random()-0.5)])
        rand_point /= np.linalg.norm(rand_point)/np.sqrt(self.rad)
        rot = sphere2rot(rand_point)
        pose = np.concatenate([rot, rand_point.reshape([3,1])], axis=1)
        pose = torch.tensor(pose, device=self.poses.device).float()
        aug_rays_ori, aug_rays_dir = get_rays(self.h, self.w, self.focal, pose)

        if self.batch_size > 0:
            n = self.h*self.w
            select_mask = torch.randperm(n)[:self.batch_size]
            batch['aug_rays_ori'] = aug_rays_ori.reshape([-1,3])[select_mask]
            batch['aug_rays_dir'] = aug_rays_dir.reshape([-1,3])[select_mask]
        else:
            batch['aug_rays_ori'] = aug_rays_ori.reshape([-1,3])
            batch['aug_rays_dir'] = aug_rays_dir.reshape([-1,3])
        return batch


def convert_rays_to_sphere_coord(rays_ori, rays_dir, r2):
    # rays_ori: [N,3]
    # rays_dir: [N,3]
    # r: radius for the sphere
    # sphere_coord: [N,3]

    # create two large enough spheres

    # http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
    # a = (x2 − x1)2 + (y2 − y1)2 + (z2 − z1)2
    # b = − 2[(x2 − x1)(xc − x1) + (y2 − y1)(yc − y1) + (z2 − z1)(zc − z1)]
    # c = (xc − x1)2 + (yc − y1)2 + (zc − z1)2 − r2
    # t = (-b±√(b^2-4ac))/2a

    # (x2-x1, y2-y1, z2-z1): rays_dir
    # (x1, y1, z1): rays_ori

    a = (rays_dir**2).sum(dim=1)
    b = -2 * (rays_dir * (-rays_ori)).sum(dim=1)
    c = (rays_ori**2).sum(dim=1) - r2
    delta = (b**2-4*a*c) > 0
    if not torch.all(delta):
        raise RuntimeError('not all deltas are positive')

    t1 = (-b+torch.sqrt(b**2-4*a*c)) / (2*a)
    t2 = (-b-torch.sqrt(b**2-4*a*c)) / (2*a)

    point1 = rays_ori+rays_dir*t1[:,None]
    point2 = rays_ori+rays_dir*t2[:,None]

    point1_dist = (point1-rays_ori).abs().sum()
    point2_dist = (point2-rays_ori).abs().sum()
    if point1_dist > point2_dist:
        point_selected = point2
    else:
        point_selected = point1
    # always use the nearest one, convert from points to coord

    # theta = np.angle(point1[:,0] + 1j*point1[:,1])
    # phi = point1[:,2]
    # sphere_coord = torch.stack([torch.tensor(theta), torch.tensor(phi)], dim=1)

    # (x,y) as coord
    sphere_coord = point_selected[:,:2]

    return sphere_coord


def sample_nearby_point_on_sphere(points, angle=np.pi/9):
    rad = torch.sqrt(points[...,0]**2+points[...,1]**2)
    full_rad = points.norm(dim=-1)
    theta = torch.atan2(rad, points[...,2])
    theta += (random.random()-0.5)*angle
    rad = full_rad * theta.sin()
    phi = torch.atan2(points[...,1], points[...,0])
    phi += (random.random()-0.5)*angle
    new_points = torch.stack([rad*phi.cos(), rad*phi.sin(), full_rad*theta.cos(),], dim=2)
    return new_points


@DATASETS.register_module()
class SyntheticSphereCoordDataset(SyntheticDataset):
    def __init__(self, *args, **kwargs):
        self.rad0 = kwargs.pop('rad0')
        self.rad1 = kwargs.pop('rad1')
        self.uv_precision = kwargs.pop('uv_precision')
        super().__init__(*args, **kwargs)

        self.all_uv = []
        for pose in self.poses:
            rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose)
            uv = convert_rays_to_sphere_coord(rays_ori.reshape([-1,3]), rays_dir.reshape([-1,3]), self.rad1)
            self.all_uv.append(self.get_unique(uv))
        self.all_uv = torch.stack(self.all_uv, dim=0)
        self.u_max, self.v_max = self.all_uv.max(0).values
        self.u_min, self.v_min = self.all_uv.min(0).values

    def get_unique(self, uv):
        uv = uv*10**self.uv_precision
        unique_uv = torch.unique(torch.round(uv).long())
        return unique_uv.float() / 10**self.uv_precision

    def __getitem__(self, idx):
        target = self.imgs[idx]
        pose = self.poses[idx, :3, :4]
        rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose)

        # aug_axis_degree = [random.random()*10 for _ in range(3)]
        # aug_mat = spatial.transform.Rotation.from_euler('zyx', aug_axis_degree, degrees=True)
        aug_axis_degree = (random.random()-0.5)*7+3
        aug_mat = spatial.transform.Rotation.from_euler('z', aug_axis_degree, degrees=True)
        aug_mat = torch.tensor(aug_mat.as_matrix()).float()
        aug_pose = aug_mat @ pose
        aug_rays_ori, aug_rays_dir = get_rays(self.h, self.w, self.focal, aug_pose)
        aug_rays_ori = sample_nearby_point_on_sphere(aug_rays_ori)

        if self.batch_size == -1:
            rays_color = target.view([-1,3])  # (N, 3)
            select_mask = torch.ones([self.h, self.w]).bool()
            rays_ori = rays_ori.view([-1,3])
            rays_dir = rays_dir.view([-1,3])
            aug_rays_ori = aug_rays_ori.view([-1,3])
            aug_rays_dir = aug_rays_dir.view([-1,3])
            return {'uv': convert_rays_to_sphere_coord(rays_ori, rays_dir, self.rad1),
                    'st': convert_rays_to_sphere_coord(rays_ori, rays_dir, self.rad0),
                    'aug_uv': convert_rays_to_sphere_coord(aug_rays_ori, aug_rays_dir, self.rad1),
                    'aug_st': convert_rays_to_sphere_coord(aug_rays_ori, aug_rays_dir, self.rad0),
                    'all_uv': self.all_uv, 
                    'h': self.h, 'w': self.w,
                    'rays_color': rays_color, }

        select_mask = torch.rand([self.h, self.w])
        if self.precrop_frac < 1:
            dH = int(self.h//2 * self.precrop_frac)
            dW = int(self.w//2 * self.precrop_frac)
            select_mask[dH:-dH, dW:-dW] *= -1

        percent_value = select_mask.view([-1]).kthvalue(self.batch_size).values
        select_mask = select_mask <= percent_value
        rays_ori = rays_ori[select_mask]  # (N, 3)
        rays_dir = rays_dir[select_mask]  # (N, 3)
        aug_rays_ori = aug_rays_ori[select_mask] 
        aug_rays_dir = aug_rays_dir[select_mask] 
        rays_color = target[select_mask]  # (N, 3)

        return {'uv': convert_rays_to_sphere_coord(rays_ori, rays_dir, self.rad1),
                'st': convert_rays_to_sphere_coord(rays_ori, rays_dir, self.rad0),
                'aug_uv': convert_rays_to_sphere_coord(aug_rays_ori, aug_rays_dir, self.rad1),
                'aug_st': convert_rays_to_sphere_coord(aug_rays_ori, aug_rays_dir, self.rad0),
                'all_uv': self.all_uv, 
                'h': self.h, 'w': self.w,
                'rays_color': rays_color, }


