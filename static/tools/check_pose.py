import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


image1 = os.path.expanduser('~/code/nerf/data/nerf/nerf_llff_data/fern/images/IMG_4026.JPG')
image2 = os.path.expanduser('~/code/nerf/data/nerf/nerf_llff_data/fern/images/IMG_4027.JPG')

# c2w, [4,4]
pose1 = np.asarray([
    [ 9.95694687e-01, -2.07959763e-02, -9.03306068e-02, -3.08100187e-01],
    [ 2.50334209e-02,  9.98626177e-01,  4.60335439e-02, 1.34677213e-01],
    [ 8.92491960e-02, -4.80966391e-02,  9.94847372e-01, 3.98987666e-02],
    [0,0,0,1],
])
# c2w, [4,4]
pose2 = np.asarray([
    [ 9.99291870e-01, -7.34605179e-03, -3.69024821e-02, -1.49857642e-01],
    [ 9.01871831e-03,  9.98929691e-01,  4.53666678e-02, 1.34402951e-01],
    [ 3.65297191e-02, -4.56673554e-02,  9.98288572e-01, 2.90271245e-02],
    [0,0,0,1],
])

h=745
w=1000
f=8.15131583e+02

K1 = np.asarray([
    [f,0,h/2],
    [0,f,w/2],
    [0,0,1],
])
K2 = K1

# K1 = np.asarray([
#     [],
#     [],
#     [],
# ])
# K2 = np.asarray([
#     [],
#     [],
#     [],
# ])

cam1_to_cam2 = np.linalg.inv(pose2) @ pose1
def map_with_depth(point, depth):
    # point [2,1]
    p = np.ones([3,1])
    p[:2,0] = point
    cam1 = np.ones([4,1])
    cam1[:3] = (np.linalg.inv(K1) @ p) * depth
    cam2 = cam1_to_cam2 @ cam1
    point2 = K2 @ cam2[:3]
    point2 /= point2[2,0]
    return point2.astype(np.int)

img1 = cv2.imread(image1, 0) 
img2 = cv2.imread(image2, 0)

sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.9*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
    if i == 50:
        break
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]

img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
for pt in pts1:
    color = tuple(np.random.randint(0,255,3).tolist())
    img1 = cv2.circle(img1, tuple(pt), 50, color, -1)
    for d in range(1,100):
        p = map_with_depth(pt, d)[:2,0]
        try:
            img2 = cv2.circle(img2, tuple(p), 5, color, -1)
        except:
            pass

fig, axes = plt.subplots(2,1)
axes[0].imshow(img1)
axes[1].imshow(img2)
fig.savefig('check.jpg')


