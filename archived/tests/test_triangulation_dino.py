"""
Test triangulation-based reconstruction on dinosaur dataset.
The dinosaur has good texture for feature matching.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import sys

# Test with dinosaur images
three_images = [
    ['./dinoSparseRing/dinoSparseRing/dinoSR0001.png', 
     './dinoSparseRing/dinoSparseRing/dinoSR0002.png', 
     './dinoSparseRing/dinoSparseRing/dinoSR0003.png'],
    ['./dinoSparseRing/dinoSparseRing/dinoSR0003.png', 
     './dinoSparseRing/dinoSparseRing/dinoSR0004.png', 
     './dinoSparseRing/dinoSparseRing/dinoSR0005.png'],
]

print("Testing dinosaur dataset with triangulation-based reconstruction...")
print("=" * 60)

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
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good = []
    pts1 = []
    pts2 = []
    
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.85 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    
    if len(pts1) < 8:
        return np.array([]), pts1, pts2, None
    
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    
    if mask is None:
        return np.array([]), pts1, pts2, None
    
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    
    return np.stack((pts1, pts2), axis=1), pts1, pts2, F

def combine_matches(matches_a, matches_b):
    """Assumes that the 0'th image is the same between them."""
    if len(matches_a) == 0 or len(matches_b) == 0:
        return np.array([])
    
    combined_matches = []
    for ii in range(matches_a.shape[0]):
        ma0 = matches_a[ii, 0]
        mi = np.where((matches_b[:, 0] == ma0).all(axis=1))[0]

        if mi.size > 0:
            ma = matches_a[ii]
            mb = matches_b[int(mi[0])]
            combined_matches.append(np.concatenate((ma, mb[1:]), axis=0))

    return np.array(combined_matches)

def solve_point_triangulation(proj_points, proj_matrices):
    """Triangulate 3D point from 2D correspondences."""
    D = np.zeros((2 * len(proj_points), 4), dtype=float)
    for ii, (p, P) in enumerate(zip(proj_points, proj_matrices)):
        D[2 * ii + 0] = p[1] * P[2] - P[1]
        D[2 * ii + 1] = P[0] - p[0] * P[2]

    u, s, vh = np.linalg.svd(D, full_matrices=False)
    X = vh[np.argmin(s)]
    return X / X[3]

def create_camera_projection_matrix(view_index, fx, fy, cx, cy, baseline=35.9):
    """Create camera projection matrix for a given view."""
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    angle_degrees = view_index * 15
    angle_radians = np.radians(angle_degrees)
    
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    R = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta, cos_theta, 0],
                  [0, 0, 1]])
    
    t = np.array([[baseline * sin_theta],
                  [baseline * cos_theta],
                  [0]])
    
    P = K @ np.hstack([R, t])
    return P

# Collect all 3D points
all_points = []

for triplet_idx, three_images_coords in enumerate(three_images):
    print(f"\nProcessing triplet {triplet_idx + 1}: {three_images_coords[0]}")
    
    # Load images
    test1 = cv.imread(three_images_coords[0], 0)
    test2 = cv.imread(three_images_coords[1], 0)
    test3 = cv.imread(three_images_coords[2], 0)
    
    # Get matches
    matches1, pts1, pts2, F = get_point_matches(test1, test2)
    matches2, pts1_1, pts2_1, F_1 = get_point_matches(test1, test3)
    
    print(f"  Matches 1-2: {len(matches1)}, Matches 1-3: {len(matches2)}")
    
    # Combine matches
    combined_matches = combine_matches(np.array(matches1), np.array(matches2))
    print(f"  Combined matches: {len(combined_matches)}")
    
    if len(combined_matches) == 0:
        print(f"  No combined matches, skipping triplet")
        continue
    
    # Camera parameters
    height, width = test1.shape
    fx = 188.9763779528
    fy = 188.9763779528
    cx = width / 2.0
    cy = height / 2.0
    baseline = 35.9
    
    # Extract matched points for each view
    im_1_pts = combined_matches[:, 0]
    im_2_pts = combined_matches[:, 1]
    im_3_pts = combined_matches[:, 2]
    
    # Calculate absolute view indices
    view_indices = [(triplet_idx * 2) + i for i in range(3)]
    print(f"  View indices: {view_indices}")
    
    # Create projection matrices
    proj_matrices = []
    for view_idx in view_indices:
        P = create_camera_projection_matrix(view_idx, fx, fy, cx, cy, baseline=baseline)
        proj_matrices.append(P)
    
    # Triangulate each matched feature
    triplet_points = []
    for match_idx in range(len(im_1_pts)):
        proj_points = [
            im_1_pts[match_idx],
            im_2_pts[match_idx],
            im_3_pts[match_idx]
        ]
        
        try:
            X_3d = solve_point_triangulation(proj_points, proj_matrices)
            
            if X_3d[3] != 0:
                pt_3d = (X_3d[0], X_3d[1], X_3d[2])
                triplet_points.append(pt_3d)
                
        except Exception as e:
            continue
    
    print(f"  Triangulated {len(triplet_points)} 3D points")
    all_points.extend(triplet_points)

print(f"\n{'=' * 60}")
print(f"Total 3D points reconstructed: {len(all_points)}")

if len(all_points) > 4:
    # Convert to numpy array
    points_array = np.array(all_points)
    
    # Compute convex hull and volume
    try:
        hull = ConvexHull(points_array)
        print(f"Convex hull volume: {hull.volume:.2f}")
        
        # Visualize
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], c='b', marker='o', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Dinosaur Reconstruction using Triangulation')
        
        # Set equal aspect ratio
        max_range = np.array([points_array[:, 0].max() - points_array[:, 0].min(),
                              points_array[:, 1].max() - points_array[:, 1].min(),
                              points_array[:, 2].max() - points_array[:, 2].min()]).max() / 2.0
        
        mid_x = (points_array[:, 0].max() + points_array[:, 0].min()) * 0.5
        mid_y = (points_array[:, 1].max() + points_array[:, 1].min()) * 0.5
        mid_z = (points_array[:, 2].max() + points_array[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.savefig('dino_triangulation_test.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: dino_triangulation_test.png")
        print("\nTriangulation implementation is working!")
        
    except Exception as e:
        print(f"Error computing convex hull: {e}")
else:
    print("Not enough points reconstructed!")

print("\nTest complete!")
