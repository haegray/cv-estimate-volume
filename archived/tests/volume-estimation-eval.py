#!/usr/bin/env python
# coding: utf-8
# Modified version for evaluation - saves plots instead of showing them

import cv2 as cv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os

# Create output directory
os.makedirs('eval_output', exist_ok=True)

def load_image(filepath):
    """Loads an image into a numpy array."""
    img = np.float32(Image.open(filepath))
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

sift = cv.SIFT_create()

def get_point_matches(img1, img2):
    """Returns matches as array: (feature track, image, coord)"""
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.85*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
    return np.stack((pts1, pts2), axis=1), pts1,pts2,F

def combine_matches(matches_a, matches_b):
    """Assumes that the 0'th image is the same between them."""
    combined_matches = []
    for ii in range(matches_a.shape[0]):
        ma0 = matches_a[ii, 0]
        mi = np.where((matches_b[:, 0] == ma0).all(axis=1))[0]
        if mi.size > 0:
            ma = matches_a[ii]
            mb = matches_b[int(mi[0])]
            combined_matches.append(np.concatenate((ma, mb[1:]), axis=0))
    return np.array(combined_matches)

def get_match_colors(image_c, combined_matches):
    colors = []
    nm = combined_matches.shape[0]
    for mi in range(nm):
        m = combined_matches[mi, 0, :]
        colors.append(image_c[m[1]-1:m[1]+2, m[0]-1:m[0]+2].sum(axis=0).sum(axis=0)/9)
    return colors

def rectify_two(test1,test2,pts1,pts2,F, path1,path2):
    h1,w1 = test1.shape
    h2,w2 = test2.shape
    _,H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1,h1))
    img1_rectified = cv.warpPerspective(test1, H1, (w1,h1))
    img2_rectified = cv.warpPerspective(test2, H2, (w2,h2))
    return img1_rectified, img2_rectified

def get_depth(img1_rectified, img2_rectified, save_name):
    block_size = 11
    min_disp = -128
    max_disp = 128
    num_disp = max_disp - min_disp
    uniquenessRatio = 5
    speckleWindowSize = 200
    speckleRange = 2
    disp12MaxDiff = 0

    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )
    
    disp = stereo.compute(np.uint8(img1_rectified), np.uint8(img2_rectified)).astype(np.float32)
    disp = cv.normalize(disp, disp, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
    disp = np.uint8(disp)

    fig = plt.figure()
    plt.imshow(disp)
    plt.savefig(f'eval_output/{save_name}_disp.png')
    plt.close()
    
    return disp

def set_points(three_images_arr, i, depth_map, save_prefix):
    if len(three_images_arr) != 3:
        raise ValueError('Array must have 3 images')
    test1 = cv.imread(three_images_arr[0],0)
    test2 = cv.imread(three_images_arr[1],0)
    test3 = cv.imread(three_images_arr[2],0)
    matches1, pts1,pts2,F = get_point_matches(test1, test2)
    matches2, pts1_1,pts2_1,F_1 = get_point_matches(test1, test3)
    
    img1_rectified, img2_rectified = rectify_two(test1,test2,pts1,pts2,F,"rect1","rect2")
    img2_rectified, img3_rectified = rectify_two(test2,test3,pts1_1,pts2_1,F_1, "rect3","rect4")

    disp1 = get_depth(img1_rectified, img2_rectified, f'{save_prefix}_depth1')
    
    combined_matches = combine_matches(np.array(matches1), np.array(matches2)) 
    colors = get_match_colors(load_image(three_images_arr[0]), combined_matches)
    return combined_matches, colors

print("=== VOLUME-ESTIMATION.PY EVALUATION ===")
print("Loading images...")
test1 = load_image('./world_rotate/trans_01.png')
test2 = load_image('./world_rotate/trans_02.png')
test3 = load_image('./world_rotate/trans_03.png')
depth_map = cv.imread('./world_rotate/depth_map_world.png')
depth_map = cv.cvtColor(depth_map, cv.COLOR_BGR2GRAY)

print(f"Image shapes: {test1.shape}, {test2.shape}, {test3.shape}")
print(f"Depth map shape: {depth_map.shape}")

three_images =  ['./world_rotate/trans_01.png','./world_rotate/trans_02.png','./world_rotate/trans_03.png']

print("Processing feature matches...")
X, colors = set_points(three_images, 0, depth_map, 'vol_est')

from scipy.spatial.transform import Rotation as R

npad = ((0, 0), (0, 0), (0, 1))
X_new = np.pad(X, pad_width=npad, mode='constant', constant_values=0)

rotation_degrees = -15
rotation_radians = np.radians(rotation_degrees)
rotation_axis = np.array([0, 0, 1])
rotation_vector = rotation_radians * rotation_axis
rotation = R.from_rotvec(rotation_vector)

print("Applying depth and rotation...")
for i,point in enumerate(X[:,0]):
    X_new[i,0,2] = depth_map[X_new[i,0,1],X_new[i,0,0]]
    X_new[i,1,2] = depth_map[X_new[i,1,1],X_new[i,1,0]]
    X_new[i,2,2] = depth_map[X_new[i,2,1],X_new[i,2,0]]
    
    X_new[i,1,:] = rotation.apply(X_new[i,1,:])
    X_new[i,2,:] = rotation.apply(X_new[i,2,:])
    X_new[i,2,:] = rotation.apply(X_new[i,2,:])

X = X_new

print(f"Number of matched features: {X.shape[0]}")
print(f"Point cloud shape: {X.shape}")

np.save("points_vol_est.npy", X)
np.save("colors_vol_est.npy", colors)

# Visualize 3D points
colors_norm = np.true_divide(colors,255)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0,0], X[:,0,1], X[:,0,2], c=colors_norm, s=1)
ax.set_xlim3d([-750, 750])
ax.set_ylim3d([-750, 750])
ax.set_zlim3d([-750, 750])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('volume-estimation.py 3D Reconstruction')
plt.savefig('eval_output/vol_est_3d.png', dpi=150)
plt.close()

print("Saved output to eval_output/vol_est_3d.png")
print("=== VOLUME-ESTIMATION.PY COMPLETE ===\n")
