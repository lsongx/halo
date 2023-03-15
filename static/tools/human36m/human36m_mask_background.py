import os
import cv2

from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = os.path.expanduser('~/code/mmdetection/configs/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco.py')
checkpoint_file = os.path.expanduser('~/data/models/trained/det/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth')
model = init_detector(config_file, checkpoint_file, device='cuda:0')

ROOT = os.path.expanduser('~/data/human36m/training/subject/processed/')
SAVE = os.path.expanduser('~/data/human36m/training/subject/masked/')
# subject_names = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
subject_names = ['S1']
action_names = ['Directions-1', 'Directions-2', 'Discussion-1', 'Discussion-2', 
                'Eating-1', 'Eating-2', 'Greeting-1', 'Greeting-2', 'Phoning-1', 
                'Phoning-2', 'Posing-1', 'Posing-2', 'Purchases-1', 'Purchases-2', 
                'Sitting-1', 'Sitting-2', 'SittingDown-1', 'SittingDown-2', 
                'Smoking-1', 'Smoking-2', 'TakingPhoto-1', 'TakingPhoto-2', 
                'Waiting-1', 'Waiting-2', 'Walking-1', 'Walking-2', 'WalkingDog-1', 
                'WalkingDog-2', 'WalkingTogether-1', 'WalkingTogether-2']
camera_names = ['54138969', '55011271', '58860488', '60457274']

i = 0
for s in subject_names:
    for a in action_names:
        for c in camera_names:
            img_base = os.path.join(ROOT, s, a, 'imageSequence-undistorted', c)
            for img_name in os.listdir(img_base):
                i += 1
                if i % 50 == 0:
                    print(i)
                frame_idx = int(img_name.split('_')[1].split('.')[0])
                write_name = f'{s}-{a}-{c}-frame{frame_idx:06d}.jpg'
                if os.path.isfile(os.path.join(SAVE, write_name)):
                    continue
                path = os.path.join(img_base, img_name)
                img = mmcv.imread(path)
                result = inference_detector(model, img)
                mask = result[1][0][0][0]
                img[~mask, :] = 255
                mmcv.imwrite(img, os.path.join(SAVE, write_name))

