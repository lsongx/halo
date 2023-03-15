# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Distribute under MIT License
# Authors:
#    - Suttisak Wizadwongsa <suttisak.w_s19[-at-]vistec.ac.th>
#    - Pakkapon Phongthawee <pakkapon.p_s19[-at-]vistec.ac.th>
#    - Jiraphon Yenphraphai <jiraphony_pro[-at-]vistec.ac.th>
#    - Supasorn Suwajanakorn <supasorn.s[-at-]vistec.ac.th>

from vcnerf.utils import get_root_logger
logger = get_root_logger()

import numpy as np
import os
import torch as pt
from collections import deque
from skimage import io
from skimage.transform import resize
import cv2
# import lpips
import imageio

import struct
import json
from scipy.spatial.transform import Rotation
import copy

# ---------------------------------------------------------------------------------------
# load_llff
# ---------------------------------------------------------------------------------------

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        logger.info(f'Minifying {r} {basedir}')

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        logger.info(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            logger.info('Removed duplicates')
        logger.info('Done')

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    shape = 5

    #poss llff arr [3, 5, images] [R | T | intrinsic]
    #intrinsic same for all images
    if os.path.isfile(os.path.join(basedir, 'hwf_cxcy.npy')):
        shape = 4
        #h, w, fx, fy, cx, cy
        intrinsic_arr = np.load(os.path.join(basedir, 'hwf_cxcy.npy'))

    poses = poses_arr[:, :-2].reshape([-1, 3, shape]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])

    if not os.path.isfile(os.path.join(basedir, 'hwf_cxcy.npy')):
        intrinsic_arr = poses[:, 4, 0]
        poses = poses[:, :4, :]

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
                    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        logger.info( f'{imgdir} does not exist, returning' )
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        logger.info( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return

    if not load_imgs:
        return poses, bds, intrinsic_arr

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    logger.info(f'Loaded image data, imgs.shape: {imgs.shape}, poses[:,-1,0]: {poses[:,-1,0]}')
    return poses, bds, imgs, intrinsic_arr

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):
    #poses [images, 3, 4] not [images, 3, 5]
    #hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center)], 1)

    return c2w

def render_path_axis(c2w, up, ax, rad, focal, N):
        render_poses = []
        center = c2w[:,3]
        hwf = c2w[:,4:5]
        v = c2w[:,ax] * rad
        for t in np.linspace(-1.,1.,N+1)[:-1]:
                c = center + t * v
                z = normalize(c - (center - focal * c2w[:,2]))
                #render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
                render_poses.append(viewmatrix(z, up, c))
        return render_poses

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    #hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        #render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def recenter_poses(poses):
    #poses [images, 3, 4]
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)

    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def spherify_poses(poses, bds):
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)

    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))

    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []

    for th in np.linspace(0.,2.*np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)

    return poses_reset, new_poses, bds

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, split_train_val = 0, render_style=''):

    # poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    poses, bds, intrinsic = _load_data(basedir, factor=factor, load_imgs=False) # factor=8 downsamples original imgs by 8x

    logger.info(f'Loaded basedir {basedir}, bds.min(){bds.min()}, bds.max(){bds.max()}')

    # Correct rotation matrix ordering and move variable dim to axis 0
    #poses [R | T] [3, 4, images]
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    #poses [3, 4, images] --> [images, 3, 4]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)

    # imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    # images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)
    else:
        c2w = poses_avg(poses)
        logger.info(f'recentered c2w.shape {c2w.shape}')

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))
        
        close_depth, inf_depth = -1, -1
        # Find a reasonable "focus depth" for this dataset
        if os.path.exists(os.path.join(basedir, 'planes_spiral.txt')):
            with open(os.path.join(basedir, 'planes_spiral.txt'), "r") as fi:
                data = [float(x) for x in fi.readline().split(" ")]
                dmin, dmax = data[:2]
                close_depth = dmin * 0.9
                inf_depth = dmax * 5.0
        # elif os.path.exists(os.path.join(basedir, 'planes.txt')):
        if os.path.exists(os.path.join(basedir, 'planes.txt')):
            with open(os.path.join(basedir, 'planes.txt'), "r") as fi:
                data = [float(x) for x in fi.readline().split(" ")]
                if len(data) ==3:
                    dmin, dmax, invz = data
                elif len(data) ==4:
                    dmin, dmax, invz, _ = data
                close_depth = dmin * 0.9
                inf_depth = dmax * 5.0

        prev_close, prev_inf = close_depth, inf_depth
        if close_depth < 0 or inf_depth < 0 or render_style == 'llff':
            close_depth, inf_depth = bds.min()*.9, bds.max()*5.

        
        if render_style == 'shiny':
            close_depth, inf_depth = bds.min()*.9, bds.max()*5.
            if close_depth < prev_close:
                close_depth = prev_close
            if inf_depth > prev_inf:
                inf_depth = prev_inf

        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
    #                         zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    render_poses = np.array(render_poses).astype(np.float32)
    if split_train_val == 0:
        # backward compatibilty

        c2w = poses_avg(poses)

        logger.info('Data:')
        # logger.info(poses.shape, images.shape, bds.shape)

        dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
        i_test = np.argmin(dists)
        logger.info(f'HOLDOUT view is i_test {i_test}')

        # images = images.astype(np.float32)
        poses = poses.astype(np.float32)

        return None, poses, bds, render_poses, i_test
    else:
        # reference_view_id should stay in train set only
        validation_ids = np.arange(poses.shape[0])
        validation_ids[::8] = -1
        validation_ids = validation_ids < 0
        train_ids = np.logical_not(validation_ids)
        train_poses = poses[train_ids]
        train_bds = bds[train_ids]
        c2w = poses_avg(train_poses)

        dists = np.sum(np.square(c2w[:3,3] - train_poses[:,:3,3]), -1)
        reference_view_id = np.argmin(dists)
        reference_depth = train_bds[reference_view_id]
        webgl = {'c2w': c2w_path,
                            'up': up,
                            'rads': rads,
                            'focal': focal,
                            'zdelta':zdelta}

        return train_poses, reference_depth, reference_view_id, render_poses, poses, intrinsic, webgl, mean_dz


# ---------------------------------------------------------------------------------------
# sfm utils
# ---------------------------------------------------------------------------------------


class SfMData:
    def __init__(self, dataset, ref_img='', scale=1, dmin=0,
        dmax=0, invz=0, render_style='', offset=200):
        self.scale = scale
        self.ref_cam = None
        self.ref_img = None
        self.render_poses = None
        self.dmin = dmin
        self.dmax = dmax
        self.invz = invz
        self.dataset = dataset
        self.dataset_type = 'unknown'
        self.render_style = render_style
        self.white_background = False #change background to white if transparent.
        self.index_split = [] #use for split dataset in blender
        self.offset = 200
        # Detect dataset type
        can_hanle = self.readDeepview(dataset) \
            or self.readLLFF(dataset, ref_img) \
            or self.readColmap(dataset) 
        if not can_hanle:
            raise Exception('Unknow dataset type')
        # Dataset processing
        self.cleanImgs()
        self.selectRef(ref_img)
        self.scaleAll(scale)
        self.selectDepth(dmin, dmax, offset)
        if self.dataset_type != 'llff': self.webgl = None

    def cleanImgs(self):
        """
        Remvoe non exist image from self.imgs
        """
        todel = []
        for image in self.imgs:
            img_path = self.dataset + "/" + self.imgs[image]['path']
            if "center" not in self.imgs[image] or not os.path.exists(img_path):
                todel.append(image)
        for it in todel:
            del self.imgs[it]

    def selectRef(self, ref_img):
        """
        Select Reference image
        """
        if ref_img == "" and self.ref_cam is not None and self.ref_img is not None:
            return
        for img_id, img in self.imgs.items():
            if ref_img in img["path"]:
                self.ref_img = img
                self.ref_cam = self.cams[img["camera_id"]]
                return
        raise Exception("reference view not found")

    def selectDepth(self, dmin, dmax, offset):
        """
        Select dmin/dmax from planes.txt / bound.txt / argparse
        """
        if self.dmin < 0 or self.dmax < 0:
            if os.path.exists(self.dataset + "/bounds.txt"):
                with open(self.dataset + "/bounds.txt", "r") as fi:
                    data = [np.reshape(np.matrix([float(y) for y in x.split(" ")]), [3, 1]) for x in fi.readlines()[3:]]
                ls = []
                for d in data:
                    v = self.ref_img['r'] * d + self.ref_img['t']
                    ls.append(v[2])
                self.dmin = np.min(ls)
                self.dmax = np.max(ls)
                self.invz = 0

            elif os.path.exists(self.dataset + "/planes.txt"):
                with open(self.dataset + "/planes.txt", "r") as fi:
                    data = [float(x) for x in fi.readline().split(" ")]
                    if len(data) == 3:
                        self.dmin, self.dmax, self.invz = data
                    elif len(data) == 2:
                        self.dmin, self.dmax = data
                    elif len(data) == 4:
                        self.dmin, self.dmax, self.invz, self.offset = data
                        self.offset = int(self.offset)
                        logger.info(f'Read offset from planes.txt: {self.offset}')
                    else:
                        raise Exception("Malform planes.txt")
            else:
                logger.info("no planes.txt or bounds.txt found")
        if dmin > 0:
            logger.info("Overriding dmin %f-> %f" % (self.dmin, dmin))
            self.dmin = dmin
        if dmax > 0:
            logger.info("Overriding dmax %f-> %f" % (self.dmax, dmax))
            self.dmax = dmax
        if offset != 200:
            logger.info(f"Overriding offset {self.offset}-> {offset}")
            self.offset = offset
        logger.info("dmin = %f, dmax = %f, invz = %d, offset = %d" % (self.dmin, self.dmax, self.invz, self.offset))

    def readLLFF(self, dataset, ref_img = ""):
        """
        Read LLFF
        Parameters:
            dataset (str): path to datasets
            ref_img (str): ref_image file name
        Returns:
            bool: return True if successful load LLFF data
        """
        if not os.path.exists(os.path.join(dataset,'poses_bounds.npy')):
            return False
        image_dir = os.path.join(dataset,'images')
        if not os.path.exists(image_dir) and not os.path.isdir(image_dir):
            return False
        # load R,T
        train_poses, reference_depth, reference_view_id, render_poses, poses, intrinsic, self.webgl, self.mean_z = load_llff_data(dataset,factor=None, split_train_val = 8, render_style = self.render_style)
        # get all image of this dataset
        images_path = [os.path.join('images', f) for f in sorted(os.listdir(image_dir))]

        #LLFF dataset has only single camera in dataset
        #H, W, focal = train_poses[reference_view_id,0,-1],train_poses[reference_view_id,1,-1],train_poses[reference_view_id,2,-1]
        if len(intrinsic) == 3:
            H, W, f = intrinsic
            cx = W / 2.0
            cy =    H / 2.0
            fx = f
            fy = f
        else:
            H, W, fx, fy, cx, cy = intrinsic


        self.cams = {0 : buildCamera(W,H,fx,fy,cx,cy) }

        # create render_poses for video render
        self.render_poses = buildNerfPoses(render_poses)

        # create imgs pytorch dataset
        # we store train and validation together
        # but it will sperate later by pytorch dataloader
        self.imgs = buildNerfPoses(poses, images_path)

        # if not set ref_cam, use LLFF ref_cam
        if ref_img == "":
            # restore image id back from reference_view_id
            # by adding missing validation index
            image_id = reference_view_id + 1 #index 0 alway in validation set
            image_id = image_id    + (image_id // 8) #every 8 will be validation set
            self.ref_cam = self.cams[0]

            self.ref_img = self.imgs[image_id] # here is reference view from train set

        # if not set dmin/dmax, use LLFF dmin/dmax
        if (self.dmin < 0 or self.dmax < 0) and (not os.path.exists(dataset + "/planes.txt")):
            self.dmin = reference_depth[0]
            self.dmax = reference_depth[1]
        self.dataset_type = 'llff'
        return True


    def scaleAll(self, scale):
        self.ocams = copy.deepcopy(self.cams) # original camera
        for cam_id in self.cams.keys():
            cam = self.cams[cam_id]
            ocam = self.ocams[cam_id]

            nw = round(ocam['width'] * scale)
            nh = round(ocam['height'] * scale)
            sw = nw / ocam['width']
            sh = nh / ocam['height']
            cam['fx'] = ocam['fx'] * sw
            cam['fy'] = ocam['fy'] * sh
            cam['px'] = (ocam['px']+0.5) * sw - 0.5
            cam['py'] = (ocam['py']+0.5) * sh - 0.5
            cam['width'] = nw
            cam['height'] = nh

    def readDeepview(self, dataset):
        if not os.path.exists(os.path.join(dataset, "models.json")):
            return False

        self.cams, self.imgs = readCameraDeepview(dataset)
        self.dataset_type = 'deepview'
        return True

    def readColmap(self, dataset):
        sparse_folder = dataset +"/dense/sparse/"
        image_folder = dataset + "/dense/images/"
        if (not os.path.exists(image_folder)) or (not os.path.exists(sparse_folder)):
            return False

        self.imgs = readImagesBinary(os.path.join(sparse_folder, "images.bin"))
        self.cams = readCamerasBinary(sparse_folder + "/cameras.bin")
        self.dataset_type = 'colmap'
        return True


def readCameraDeepview(dataset):
    cams = {}
    imgs = {}
    with open(os.path.join(dataset, "models.json"), "r") as fi:
        js = json.load(fi)
        for i, cam in enumerate(js):
            for j, cam_info in enumerate(cam):
                img_id = cam_info['relative_path']
                cam_id = img_id.split('/')[0]

                rotation = Rotation.from_rotvec(np.float32(cam_info['orientation'])).as_matrix().astype(np.float32)
                position = np.array([cam_info['position']], dtype='f').reshape(3, 1)

                if i == 0:
                    cams[cam_id] = {
                        'width': int(cam_info['width']),
                        'height': int(cam_info['height']),
                        'fx': cam_info['focal_length'],
                        'fy': cam_info['focal_length'] * cam_info['pixel_aspect_ratio'],
                        'px': cam_info['principal_point'][0],
                        'py': cam_info['principal_point'][1]
                    }
                imgs[img_id] = {
                    "camera_id": cam_id,
                    "r": rotation,
                    "t": -np.matmul(rotation, position),
                    "R": rotation.transpose(),
                    "center": position,
                    "path": cam_info['relative_path']
                }
    return cams, imgs

def readImagesBinary(path):
    images = {}
    f = open(path, "rb")
    num_reg_images = struct.unpack('Q', f.read(8))[0]
    for i in range(num_reg_images):
        image_id = struct.unpack('I', f.read(4))[0]
        qv = np.fromfile(f, np.double, 4)

        tv = np.fromfile(f, np.double, 3)
        camera_id = struct.unpack('I', f.read(4))[0]

        name = ""
        name_char = -1
        while name_char != b'\x00':
            name_char = f.read(1)
            if name_char != b'\x00':
                name += name_char.decode("ascii")


        num_points2D = struct.unpack('Q', f.read(8))[0]

        for i in range(num_points2D):
            f.read(8 * 2) # for x and y
            f.read(8) # for point3d Iid

        r = Rotation.from_quat([qv[1], qv[2], qv[3], qv[0]]).as_dcm().astype(np.float32)
        t = tv.astype(np.float32).reshape(3, 1)

        R = np.transpose(r)
        center = -R @ t
        # storage is scalar first, from_quat takes scalar last.
        images[image_id] = {
            "camera_id": camera_id,
            "r": r,
            "t": t,
            "R": R,
            "center": center,
            "path": "dense/images/" + name
        }

    f.close()
    return images

def readCamerasBinary(path):
    cams = {}
    f = open(path, "rb")
    num_cameras = struct.unpack('Q', f.read(8))[0]

    # becomes pinhole camera model , 4 parameters
    for i in range(num_cameras):
        camera_id = struct.unpack('I', f.read(4))[0]
        model_id = struct.unpack('i', f.read(4))[0]

        width = struct.unpack('Q', f.read(8))[0]
        height = struct.unpack('Q', f.read(8))[0]

        fx = struct.unpack('d', f.read(8))[0]
        fy = struct.unpack('d', f.read(8))[0]
        px = struct.unpack('d', f.read(8))[0]
        py = struct.unpack('d', f.read(8))[0]

        cams[camera_id] = {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "px": px,
            "py": py
        }
        # fx, fy, cx, cy
    f.close()
    return cams

def nerf_pose_to_ours(cam):
    R = cam[:3, :3]
    center = cam[:3, 3].reshape([3,1])
    center[1:] *= -1
    R[1:, 0] *= -1
    R[0, 1:] *= -1

    r = np.transpose(R)
    t = -r @ center
    return R, center, r, t

def buildCamera(W,H,fx,fy,cx,cy):
    return {
        "width": int(W),
        "height": int(H),
        "fx": float(fx),
        "fy": float(fy),
        "px": float(cx),
        "py": float(cy)
    }

def buildNerfPoses(poses, images_path = None):
    output = {}
    for poses_id in range(poses.shape[0]):
        R, center, r, t = nerf_pose_to_ours(poses[poses_id].astype(np.float32))
        output[poses_id] = {
            "camera_id": 0,
            "r": r,
            "t": t,
            "R": R,
            "center": center,
            "ori_pose": poses[poses_id]
        }
        if images_path is not None:
            output[poses_id]["path"] = images_path[poses_id]

    return output



class OrbiterDataset:
    def __init__(self, dataset, ref_img, scale, dmin,
            dmax, invz, transform=None, cache_size=50,
            render_style='', offset=200, cv2resize=False):
        self.scale = scale
        self.dataset = dataset
        self.transform = transform
        self.sfm = SfMData(dataset,
                           ref_img=ref_img,
                           dmin=dmin,
                           dmax=dmax,
                           invz=invz,
                           scale=scale,
                           render_style=render_style,
                           offset = offset)

        self.sfm.ref_rT = pt.from_numpy(self.sfm.ref_img['r']).t()
        self.sfm.ref_t = pt.from_numpy(self.sfm.ref_img['t'])
        self.cv2resize = cv2resize

        self.imgs = []
        self.poses = []

        self.ref_id = -1
        for i, ind in enumerate(self.sfm.imgs):
            img = self.sfm.imgs[ind]
            self.imgs.append(img)
            if ref_img in img['path']:
                self.ref_id = len(self.imgs) - 1

        self.cache = {}
        self.cache_queue = deque()
        self.cache_size = cache_size


    def fromCache(self, img_path, scale):
        p = (img_path, scale)
        if p in self.cache:
            return self.cache[p]

        img = io.imread(img_path)

        if img.shape[2] > 3 and self.sfm.white_background:
            img[img[:, :, 3] == 0] = [255, 255, 255, 0]

        img = img[:, :, :3]

        if scale != 1:
            h, w = img.shape[:2]
            if self.sfm.dataset_type == 'deepview':
                newh = int(h * scale) #always floor down height
                neww = round(w * scale)
            else:
                newh = round(h * scale)
                neww = round(w * scale)
                
            if self.cv2resize:
                img = cv2.resize(img, (neww, newh),interpolation=cv2.INTER_AREA)
            else:
                img = resize(img, (newh, neww))

        if len(self.cache) == self.cache_size:
            dp = self.cache_queue.popleft()
            del self.cache[dp]

        self.cache_queue.append(p)
        self.cache[p] = img
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if pt.is_tensor(idx):
                idx = idx.tolist()

        img = self.fromCache(self.dataset + "/" + self.imgs[idx]['path'], self.scale)

        img = np.transpose(img, [2, 0, 1]).astype(np.float32)
        if np.max(img) > 1:
            img /= 255.0

        im = self.imgs[idx]
        cam = self.sfm.cams[im['camera_id']]
        feature = {
            'image': img,
            'height': img.shape[1],
            'width': img.shape[2],
            'r': im['r'],
            't': im['t'],
            'center':im['center'],
            'ori_pose':im['ori_pose'],
            'fx': cam['fx'],
            'fy': cam['fy'],
            'px': cam['px'],
            'py': cam['py'],
            'path': im['path']
        }

        return feature

