import cv2
import os
import glob
import imageio
from scipy.interpolate import CubicSpline

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from vcnerf.utils import get_root_logger


def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    return transforms.Compose(ops)


def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )


def quat_to_rot(q):
    """
    Quaternion to rotation matrix
    """
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3), device=q.device)
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return R


class DVRDataset(torch.utils.data.Dataset):
    """
    Dataset from DVR (Niemeyer et al. 2020)
    Provides 3D-R2N2 and NMR renderings
    """

    def __init__(
        self,
        path,
        stage="val",
        load_object=None,
        list_prefix="new_",
        image_size=None,
        sub_format="dtu",
        scale_focal=False,
        max_imgs=49,
        z_near=0.1,
        z_far=5.0,
        skip_step=None,
        num_views=40,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
        :param sub_format shapenet | dtu dataset sub-type.
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()
        self.base_path = path
        assert os.path.exists(self.base_path)

        cats = [x for x in glob.glob(os.path.join(path, "*")) if os.path.isdir(x)]

        if stage == "train":
            file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        elif stage == "val":
            file_lists = [os.path.join(x, list_prefix + "val.lst") for x in cats]
        elif stage == "test":
            file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]

        all_objs = []
        for file_list in file_lists:
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)
        if load_object is not None:
            all_objs = [(cat, os.path.join(base_dir, load_object)),]

        self.all_objs = all_objs
        self.stage = stage

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        print(
            "Loading DVR dataset",
            self.base_path,
            "stage",
            stage,
            len(self.all_objs),
            "objs",
            "type:",
            sub_format,
            "names:",
            self.all_objs
        )

        self.image_size = image_size
        if sub_format == "dtu":
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        self.sub_format = sub_format
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False
        self.render_poses = self.gen_render_poses(num_views)

    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        cat, root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob.glob(os.path.join(root_dir, "image", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob.glob(os.path.join(root_dir, "mask", "*.png")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]
            mask_paths = [mask_paths[i] for i in sel_indices]

        cam_path = os.path.join(root_dir, "cameras.npz")
        all_cam = np.load(cam_path)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        focal = None
        if self.sub_format != "shapenet":
            # Prepare to average intrinsics over images
            fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0

        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            if self.scale_focal:
                x_scale = img.shape[1] / 2.0
                y_scale = img.shape[0] / 2.0
                xy_delta = 1.0
            else:
                x_scale = y_scale = 1.0
                xy_delta = 0.0

            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]
            if self.sub_format == "dtu":
                # Decompose projection matrix
                # DVR uses slightly different format for DTU set
                P = all_cam["world_mat_" + str(i)]
                P = P[:3]

                K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
                K = K / K[2, 2]

                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R.transpose()
                pose[:3, 3] = (t[:3] / t[3])[:, 0]

                scale_mtx = all_cam.get("scale_mat_" + str(i))
                if scale_mtx is not None:
                    norm_trans = scale_mtx[:3, 3:]
                    norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

                    pose[:3, 3:] -= norm_trans
                    pose[:3, 3:] /= norm_scale

                fx += torch.tensor(K[0, 0]) * x_scale
                fy += torch.tensor(K[1, 1]) * y_scale
                cx += (torch.tensor(K[0, 2]) + xy_delta) * x_scale
                cy += (torch.tensor(K[1, 2]) + xy_delta) * y_scale
            else:
                # ShapeNet
                wmat_inv_key = "world_mat_inv_" + str(i)
                wmat_key = "world_mat_" + str(i)
                if wmat_inv_key in all_cam:
                    extr_inv_mtx = all_cam[wmat_inv_key]
                else:
                    extr_inv_mtx = all_cam[wmat_key]
                    if extr_inv_mtx.shape[0] == 3:
                        extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
                    extr_inv_mtx = np.linalg.inv(extr_inv_mtx)

                intr_mtx = all_cam["camera_mat_" + str(i)]
                fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
                assert abs(fx - fy) < 1e-9
                fx = fx * x_scale
                if focal is None:
                    focal = fx
                else:
                    assert abs(fx - focal) < 1e-5
                pose = extr_inv_mtx

            pose = (
                self._coord_trans_world
                @ torch.tensor(pose, dtype=torch.float32)
                @ self._coord_trans_cam
            )

            img_tensor = self.image_to_tensor(img)
            if mask_path is not None:
                mask_tensor = self.mask_to_tensor(mask)

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) == 0:
                    raise RuntimeError(
                        "ERROR: Bad image at", rgb_path, "please investigate!"
                    )
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                all_masks.append(mask_tensor)
                all_bboxes.append(bbox)

            all_imgs.append(img_tensor)
            all_poses.append(pose)

        if self.sub_format != "shapenet":
            fx /= len(rgb_paths)
            fy /= len(rgb_paths)
            cx /= len(rgb_paths)
            cy /= len(rgb_paths)
            focal = torch.tensor((fx, fy), dtype=torch.float32)
            c = torch.tensor((cx, cy), dtype=torch.float32)
            all_bboxes = None
        elif mask_path is not None:
            all_bboxes = torch.stack(all_bboxes)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks)
        else:
            all_masks = None

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            if self.sub_format != "shapenet":
                c *= scale
            elif mask_path is not None:
                all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            if all_masks is not None:
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
        }
        if all_masks is not None:
            result["masks"] = all_masks
        if self.sub_format != "shapenet":
            result["c"] = c
        else:
            result["bbox"] = all_bboxes
        return result

    def gen_render_poses(self, num_views):
        t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
        pose_quat = torch.tensor(
            [
                [0.9698, 0.2121, 0.1203, -0.0039],
                [0.7020, 0.1578, 0.4525, 0.5268],
                [0.6766, 0.3176, 0.5179, 0.4161],
                [0.9085, 0.4020, 0.1139, -0.0025],
                [0.9698, 0.2121, 0.1203, -0.0039],
            ]
        )
        n_inter = num_views // 5
        num_views = n_inter * 5
        t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
        scales = np.array([2.0, 2.0, 2.0, 2.0, 2.0]).astype(np.float32)

        s_new = CubicSpline(t_in, scales, bc_type="periodic")
        s_new = s_new(t_out)

        q_new = CubicSpline(t_in, pose_quat.detach().cpu().numpy(), bc_type="periodic")
        q_new = q_new(t_out)
        q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
        q_new = torch.from_numpy(q_new).float()

        render_poses = []
        for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
            new_q = new_q.unsqueeze(0)
            R = quat_to_rot(new_q)
            t = R[:, :, 2] * scale
            new_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            new_pose[:, :3, :3] = R
            new_pose[:, :3, 3] = t
            render_poses.append(new_pose)
        render_poses = torch.cat(render_poses, dim=0)
        return render_poses
