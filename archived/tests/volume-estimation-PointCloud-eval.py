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
    
    return np.stack((pts1, pts2), axis=1)

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

def correct_affine_ambiguity(M, S):
    Atilde = M
    Xtilde = S
    m = Atilde.shape[0]//2
    Am = np.zeros((3*m, 9))
    bm = np.zeros(3*m)
    for mi in range(m):
        Atl = Atilde[2*mi:2*mi+2]
        for ri in range(3):
            for ci in range(3):
                Am[3*mi+0, 3*ri + ci] = Atl[0, ri] * Atl[0, ci]
                bm[3*mi+0] = 1
                Am[3*mi+1, 3*ri + ci] = Atl[1, ri] * Atl[1, ci]
                bm[3*mi+1] = 1
                Am[3*mi+2, 3*ri + ci] = Atl[0, ri] * Atl[1, ci]
                bm[3*mi+2] = 0

    CCT = np.reshape(np.linalg.lstsq(Am, bm, rcond=None)[0], (3, 3))
    C = 0.5*(CCT + CCT.T)
    w, v = np.linalg.eig(C)
    CCT = v @ np.diag(np.abs(w)) @ np.linalg.inv(v)
    C = np.linalg.cholesky(CCT)
    
    A = Atilde @ C
    X = np.linalg.inv(C) @ Xtilde
    return A, X

def reproject_image_points(A, X, centers):  
    mtx = A @ X
    new_matrix = np.zeros((mtx.shape[1],int(mtx.shape[0]/2),2))
    for j in range(mtx.shape[1]):
        new_matrix[j][0] = [mtx[0][j]+centers[0][0],mtx[1][j]+centers[0][1]]
        new_matrix[j][1] = [mtx[2][j]+centers[1][0],mtx[3][j]+centers[1][1]]
        new_matrix[j][2] = [mtx[4][j]+centers[2][0],mtx[5][j]+centers[2][1]]
    return new_matrix

print("=== VOLUME-ESTIMATION-POINTCLOUD-MANYIMAGES.PY EVALUATION ===")
print("Loading images...")
test1 = cv.imread('./world_rotate/trans_01.png',0)
test2 = cv.imread('./world_rotate/trans_02.png',0)
test3 = cv.imread('./world_rotate/trans_03.png',0)

print(f"Image shapes: {test1.shape}, {test2.shape}, {test3.shape}")

print("Matching features...")
matches1 = get_point_matches(test1, test2)
matches2 = get_point_matches(test1, test3)

print(f"Matches 1-2: {matches1.shape[0]}, Matches 1-3: {matches2.shape[0]}")

combined_matches = combine_matches(matches1, matches2) 
print(f"Combined matches: {combined_matches.shape[0]}")

colors = get_match_colors(load_image('./world_rotate/trans_01.png'), combined_matches)

centers = np.mean(combined_matches, axis=0)
matches_norm = combined_matches - centers
f_points, im_num, f_coords = matches_norm.shape

print("Building measurement matrix...")
D = np.zeros((6,f_points))
for i in range(0,f_points):
    D[0][i] = matches_norm[i][0][0]
    D[1][i] = matches_norm[i][0][1]
    D[2][i] = matches_norm[i][1][0]
    D[3][i] = matches_norm[i][1][1]
    D[4][i] = matches_norm[i][2][0]
    D[5][i] = matches_norm[i][2][1]

print("Performing SVD factorization...")
[U,S,V] = np.linalg.svd(D)
true_s = np.zeros((U.shape[1], V.shape[0]))
true_s[:S.size, :S.size] = np.diag(S)

U_3col = np.zeros((U.shape[1],3))
for i in range(U.shape[0]):
    U_3col[i][0] = U[i][0]
    U_3col[i][1] = U[i][1]
    U_3col[i][2] = U[i][2]
U = U_3col

W = np.zeros((3,3))
for i in range(0,3):
    for j in range(0,3):
        W[i][j] = true_s[i][j]

V_3row = np.zeros((3,V.shape[0]))
V_3row[0] = V[0]
V_3row[1] = V[1]
V_3row[2] = V[2]
V = V_3row

M = U@np.sqrt(W)
S = np.sqrt(W) @ V

print("Correcting affine ambiguity...")
A, X = correct_affine_ambiguity(M, S)
reprojected_image_points = reproject_image_points(A, X, centers)

print(f"Reconstructed 3D points shape: {X.shape}")
print(f"Number of 3D points: {X.shape[1]}")

# Save visualization
fig = plt.figure(figsize=(8, 8), dpi=150)
stacked_images = (test1, test2, test3)
num_images = len(stacked_images)
for ii in range(num_images):
    plt.subplot(2, 2, ii+1)
    plt.imshow(stacked_images[ii], cmap='gray')
    plt.plot(combined_matches[:, ii, 0], combined_matches[:, ii, 1], '.')
    plt.plot(reprojected_image_points[:, ii, 0], reprojected_image_points[:, ii, 1], 'ro', markerfacecolor='none')
    plt.title(f'Image {ii+1}')
plt.tight_layout()
plt.savefig('eval_output/pointcloud_reprojection.png')
plt.close()

np.save("points_pointcloud.npy", X)
np.save("colors_pointcloud.npy", colors)

# Visualize 3D points
colors_norm = np.true_divide(colors,255)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[0, :], X[1, :], X[2, :], c=colors_norm, s=1)
ax.set_xlim3d([-250, 250])
ax.set_ylim3d([-250, 250])
ax.set_zlim3d([-250, 250])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('PointCloud-ManyImages 3D Reconstruction')
plt.savefig('eval_output/pointcloud_3d.png', dpi=150)
plt.close()

print("Saved outputs to eval_output/")
print("=== VOLUME-ESTIMATION-POINTCLOUD-MANYIMAGES.PY COMPLETE ===\n")
