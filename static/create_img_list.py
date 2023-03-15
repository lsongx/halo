import os

img_root = os.path.expanduser('~/data/3d/rigger/megan_joyfuljump')
frame_start = 9
frame_end = 20
cam_start = 0
cam_end = 19

imgs = []
for cam in range(cam_start, cam_end):
    for frame in range(frame_start, frame_end):
        imgs.append(f'cam{cam:02d}_frame{frame:04d}.png cam{cam:02d}.json cam{cam:02d}_frame{frame:04d}_pose.json')

with open(os.path.join(img_root, 'image_list.txt'), 'w') as f:
    f.write('\n'.join(imgs))
