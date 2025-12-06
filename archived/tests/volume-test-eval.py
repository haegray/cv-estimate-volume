#!/usr/bin/env python
# Modified version for evaluation - saves plots instead of showing them

import cv2 as cv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os

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
    
    pt1 = np.append(pts1[0],1)
    pt2 = np.append(pts2[0],1)
    
    return np.stack((pts1, pts2), axis=1), pts1, pts2, F

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

def rectify_two(test1,test2,pts1,pts2, F):
    h1,w1 = test1.shape
    h2,w2 = test2.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1,h1))
    img1_rectified = cv.warpPerspective(test1, H1, (w1,h1))
    img2_rectified = cv.warpPerspective(test2, H2, (w2,h2))
    return img1_rectified, img2_rectified

def set_points(three_images_arr, i, depth_map):
    if len(three_images_arr) != 3:
        raise ValueError('Array must have 3 images')
    test1 = cv.imread(three_images_arr[0],0)
    test2 = cv.imread(three_images_arr[1],0)
    test3 = cv.imread(three_images_arr[2],0)
    matches1, pts1,pts2,F = get_point_matches(test1, test2)
    matches2, pts1_1,pts2_1,F_1 = get_point_matches(test1, test3)
    
    img1_rectified, img2_rectified = rectify_two(test1,test2,pts1,pts2, F)
    img1_rectified, img3_rectified = rectify_two(test1,test3,pts1_1,pts2_1, F_1)
    
    combined_matches = combine_matches(np.array(matches1), np.array(matches2)) 
    colors = get_match_colors(load_image(three_images_arr[0]), combined_matches)
    return combined_matches, colors, img1_rectified, img2_rectified, img3_rectified 

print("=== VOLUME-TEST.PY EVALUATION ===")
print("Processing 12 image triplets...")

depth_map = cv.imread('./world_rotate/depth_map_world.png')
depth_map = cv.cvtColor(depth_map, cv.COLOR_BGR2GRAY)

ulti_x = []
ulti_y = []
ulti_z = []

three_images =  [['./world_rotate/trans_01.png','./world_rotate/trans_02.png','./world_rotate/trans_03.png'],
                ['./world_rotate/trans_03.png','./world_rotate/trans_04.png','./world_rotate/trans_05.png'],
                 ['./world_rotate/trans_05.png','./world_rotate/trans_06.png','./world_rotate/trans_07.png'],
                 ['./world_rotate/trans_07.png','./world_rotate/trans_08.png','./world_rotate/trans_09.png'],
                 ['./world_rotate/trans_09.png','./world_rotate/trans_10.png','./world_rotate/trans_11.png'],
                 ['./world_rotate/trans_11.png','./world_rotate/trans_12.png','./world_rotate/trans_13.png'],
                 ['./world_rotate/trans_13.png','./world_rotate/trans_14.png','./world_rotate/trans_15.png'],
                 ['./world_rotate/trans_15.png','./world_rotate/trans_16.png','./world_rotate/trans_17.png'],
                 ['./world_rotate/trans_17.png','./world_rotate/trans_18.png','./world_rotate/trans_19.png'],
                 ['./world_rotate/trans_19.png','./world_rotate/trans_20.png','./world_rotate/trans_21.png'],
                 ['./world_rotate/trans_21.png','./world_rotate/trans_22.png','./world_rotate/trans_23.png'],
                 ['./world_rotate/trans_23.png','./world_rotate/trans_24.png','./world_rotate/trans_25.png']]

for cmap_option, three_images_coords in enumerate(three_images):
    print(f"Processing triplet {cmap_option+1}/12...")
    X_matches, colors, img1_rectified, img2_rectified, img3_rectified = set_points(three_images_coords, 0, depth_map)

    b = 35.9 
    f = 188.9 

    test1 = cv.imread(three_images_coords[0],0)
    width = test1.shape[0]
    height = test1.shape[1]

    from skimage import filters
    src = cv.imread('./world_rotate/depthmap.png',0)
    dim = (width, height)
    src = cv.resize(src, dim, interpolation = cv.INTER_AREA)
    height, width = src.shape

    x_center = src.shape[0]
    y_center = src.shape[1]
    foc_l = 188.9763779528

    im_1_pts = X_matches[:,0]
    im_2_pts = X_matches[:,1]
    im_3_pts = X_matches[:,2]

    ims = [im_1_pts, im_2_pts, im_3_pts]

    img_pts = []
    for imj in ims:
        X_s = []
        Z_s = []
        Y_s = []

        u_coords = imj[:,0]
        v_coords = imj[:,1]
        for u,v in zip(u_coords, v_coords):
                Y = src[u][v]*foc_l
                Z = (v - x_center) * src[u][v] / (src[u][v] / foc_l)
                X = (u - y_center) * src[u][v] / (src[u][v] / foc_l)
                Z_s.append(Z)
                X_s.append(X)
                Y_s.append(Y)
        img_pts.append(list(zip(X_s, Y_s, Z_s)))

    from scipy.spatial.transform import Rotation as R

    for i in range(0,len(img_pts)):
        angle = -15*((cmap_option*2) + i)
        rotation_degrees = angle
        rotation_radians = np.radians(rotation_degrees)
        rotation_axis = np.array([0, 0, 1])
        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)

        for j,pt in enumerate(img_pts[i]):
            if(cmap_option == 0 and j == 0):
                continue
            img_pts[i][j] = rotation.apply(pt)

    im_1_pts = img_pts[0]
    im_2_pts = img_pts[1]
    im_3_pts = img_pts[2]

    X_s, Y_s, Z_s = [a_tuple[0] for a_tuple in im_1_pts], [a_tuple[1] for a_tuple in im_1_pts], [a_tuple[2] for a_tuple in im_1_pts]
    X_s_1, Y_s_1, Z_s_1 = [a_tuple[0] for a_tuple in im_2_pts], [a_tuple[1] for a_tuple in im_2_pts], [a_tuple[2] for a_tuple in im_2_pts]
    X_s_2, Y_s_2, Z_s_2 = [a_tuple[0] for a_tuple in im_3_pts], [a_tuple[1] for a_tuple in im_3_pts], [a_tuple[2] for a_tuple in im_3_pts]
    
    for x,x1,x2 in zip(X_s,X_s_1,X_s_2):
        ulti_x.append(x)
        ulti_x.append(x1)
        ulti_x.append(x2)
    
    for y,y1,y2 in zip(Y_s,Y_s_1,Y_s_2):
        ulti_y.append(y)
        ulti_y.append(y1)
        ulti_y.append(y2)

    for z,z1,z2 in zip(Z_s,Z_s_1,Z_s_2):
        ulti_z.append(z)
        ulti_z.append(z1)
        ulti_z.append(z2)

ulti_pts = np.array(list(zip(ulti_x,ulti_y,ulti_z)))
print(f"Total points collected: {ulti_pts.shape[0]}")

from scipy.spatial import ConvexHull

hull = ConvexHull(ulti_pts)
indices = hull.simplices
faces = ulti_pts[indices]

print(f'Hull volume: {hull.volume}')

from operator import itemgetter

def triangleVol(p1, p2, p3):
    pts = [p1,p2,p3]
    p1 = max(pts, key=itemgetter(2))
    new_pts = []
    for i in pts:
        if i[0] != p1[0] or i[1] != p1[1] or i[2] != p1[2]:
            new_pts.append(i)
    pts = new_pts
    N = np.linalg.norm(np.cross((pts[0] - p1), (pts[1] - p1)));
    PV = np.linalg.norm([0,0,0] - p1);
    if(np.dot( PV, N ) > 0.0 ):
        p2 = pts[0]
        p3 = pts[1]
    else:
        p2 = pts[1]
        p3 = pts[0]
    
    v321 = p3[0]*p2[1]*p1[2];
    v231 = p2[0]*p3[1]*p1[2];
    v312 = p3[0]*p1[1]*p2[2];
    v132 = p1[0]*p3[1]*p2[2];
    v213 = p2[0]*p1[1]*p3[2];
    v123 = p1[0]*p2[1]*p3[2];
    return (1.0/6.0)*(-v321 + v231 + v312 - v132 - v213 + v123);

def meshVol(faces):
    vols = []
    for t in faces:
        vols.append(triangleVol(t[0], t[1], t[2]));
    return abs(sum(vols));

calc_vol = meshVol(faces)
print(f"Calculated volume (meshVol): {calc_vol}")
print(f"Difference in volume: {abs(hull.volume - calc_vol)}")

# Save 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ulti_pts[:,0], ulti_pts[:,1], ulti_pts[:,2], s=1, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title(f'volume-test.py 3D Reconstruction\nHull Volume: {hull.volume:.2f}')
plt.savefig('eval_output/volume_test_3d.png', dpi=150)
plt.close()

print("Saved output to eval_output/volume_test_3d.png")
print("=== VOLUME-TEST.PY COMPLETE ===\n")
