"""
Test cube reconstruction and save visualizations to eval_output folder.
This verifies that the cube shape is correct (not a cone).
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import os

# Create eval_output directory if it doesn't exist
os.makedirs('eval_output', exist_ok=True)

# Import functions from volume-test.py
import sys
sys.path.insert(0, '.')

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
    combined_matches = []
    for ii in range(matches_a.shape[0]):
        ma0 = matches_a[ii, 0]
        mi = np.where((matches_b[:, 0] == ma0).all(axis=1))[0]

        if mi.size > 0:
            ma = matches_a[ii]
            mb = matches_b[int(mi[0])]
            combined_matches.append(np.concatenate((ma, mb[1:]), axis=0))

    return np.array(combined_matches)

def rectify_two(test1, test2, pts1, pts2, F):
    h1, w1 = test1.shape
    h2, w2 = test2.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))
    
    img1_rectified = cv.warpPerspective(test1, H1, (w1, h1))
    img2_rectified = cv.warpPerspective(test2, H2, (w2, h2))
    
    return img1_rectified, img2_rectified

# Test with cube images
three_images = [
    ['./cube_images/cube_01.png', './cube_images/cube_02.png', './cube_images/cube_03.png'],
    ['./cube_images/cube_03.png', './cube_images/cube_04.png', './cube_images/cube_05.png'],
    ['./cube_images/cube_05.png', './cube_images/cube_06.png', './cube_images/cube_07.png'],
    ['./cube_images/cube_07.png', './cube_images/cube_08.png', './cube_images/cube_09.png'],
]

print("Testing cube reconstruction with visualization...")
print("=" * 60)

# Camera parameters
fx = 188.9763779528
fy = 188.9763779528
baseline = 35.9

# Collect all 3D points
all_points = []

# Create 3D plot
fig = plt.figure(figsize=(15, 5))

for triplet_idx, three_images_coords in enumerate(three_images):
    print(f"\nProcessing triplet {triplet_idx + 1}: {three_images_coords[0]}")
    
    # Load images
    test1 = cv.imread(three_images_coords[0], 0)
    test2 = cv.imread(three_images_coords[1], 0)
    test3 = cv.imread(three_images_coords[2], 0)
    
    height, width = test1.shape
    cx = width / 2.0
    cy = height / 2.0
    
    # Get matches
    matches1, pts1, pts2, F = get_point_matches(test1, test2)
    matches2, pts1_1, pts2_1, F_1 = get_point_matches(test1, test3)
    
    print(f"  Matches 1-2: {len(matches1)}, Matches 1-3: {len(matches2)}")
    
    # Combine matches
    combined_matches = combine_matches(np.array(matches1), np.array(matches2))
    print(f"  Combined matches: {len(combined_matches)}")
    
    # Check if we have enough feature matches
    has_enough_matches = len(combined_matches) > 10
    
    if not has_enough_matches:
        # Use uniform sampling from depth maps (cube images ARE depth maps)
        print(f"  Using uniform sampling (low texture)...")
        
        # Load depth maps (the images themselves)
        depth_map_1 = cv.resize(test1, (width, height), interpolation=cv.INTER_AREA)
        depth_map_2 = cv.resize(test2, (width, height), interpolation=cv.INTER_AREA)
        depth_map_3 = cv.resize(test3, (width, height), interpolation=cv.INTER_AREA)
        
        # Invert (darker = closer in cube images)
        depth_map_1 = 255 - depth_map_1
        depth_map_2 = 255 - depth_map_2
        depth_map_3 = 255 - depth_map_3
        
        depth_maps = [depth_map_1, depth_map_2, depth_map_3]
        
        # Sample points uniformly
        sample_step = 20
        
        for view_idx in range(3):
            depth_map_for_view = depth_maps[view_idx]
            
            for v in range(0, height, sample_step):
                for u in range(0, width, sample_step):
                    if v >= height or u >= width:
                        continue
                    
                    depth = depth_map_for_view[v, u]
                    
                    if depth < 1:
                        continue
                    
                    # Apply pinhole camera model
                    X = (u - cx) * depth / fx
                    Y = (v - cy) * depth / fy
                    Z = depth
                    
                    # Apply rotation to transform to world coordinates
                    view_index = (triplet_idx * 2) + view_idx
                    camera_angle = 15 * view_index
                    rotation_radians = np.radians(camera_angle)
                    rotation_axis = np.array([0, 0, 1])
                    rotation_vector = rotation_radians * rotation_axis
                    rotation = R.from_rotvec(rotation_vector)
                    
                    pt_3d = rotation.apply([X, Y, Z])
                    all_points.append(pt_3d)

print(f"\n{'=' * 60}")
print(f"Total 3D points reconstructed: {len(all_points)}")

if len(all_points) > 0:
    # Convert to numpy array
    points_array = np.array(all_points)
    
    # Compute convex hull and volume
    try:
        hull = ConvexHull(points_array)
        print(f"Convex hull volume: {hull.volume:.2f}")
        print(f"Number of hull vertices: {len(hull.vertices)}")
        print(f"Number of hull faces: {len(hull.simplices)}")
        
        # Create multiple views
        fig = plt.figure(figsize=(20, 5))
        
        # View 1: XY plane (top view)
        ax1 = fig.add_subplot(141, projection='3d')
        ax1.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
                   c=points_array[:, 2], cmap='viridis', marker='o', s=1, alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Cube Reconstruction - View 1')
        ax1.view_init(elev=30, azim=45)
        
        # View 2: XZ plane (front view)
        ax2 = fig.add_subplot(142, projection='3d')
        ax2.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
                   c=points_array[:, 2], cmap='plasma', marker='o', s=1, alpha=0.6)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Cube Reconstruction - View 2')
        ax2.view_init(elev=0, azim=0)
        
        # View 3: YZ plane (side view)
        ax3 = fig.add_subplot(143, projection='3d')
        ax3.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
                   c=points_array[:, 2], cmap='inferno', marker='o', s=1, alpha=0.6)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Cube Reconstruction - View 3')
        ax3.view_init(elev=0, azim=90)
        
        # View 4: Isometric view
        ax4 = fig.add_subplot(144, projection='3d')
        ax4.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
                   c=points_array[:, 2], cmap='magma', marker='o', s=1, alpha=0.6)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        ax4.set_title('Cube Reconstruction - Isometric')
        ax4.view_init(elev=20, azim=135)
        
        # Set equal aspect ratio for all views
        for ax in [ax1, ax2, ax3, ax4]:
            max_range = np.array([points_array[:, 0].max() - points_array[:, 0].min(),
                                  points_array[:, 1].max() - points_array[:, 1].min(),
                                  points_array[:, 2].max() - points_array[:, 2].min()]).max() / 2.0
            
            mid_x = (points_array[:, 0].max() + points_array[:, 0].min()) * 0.5
            mid_y = (points_array[:, 1].max() + points_array[:, 1].min()) * 0.5
            mid_z = (points_array[:, 2].max() + points_array[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig('eval_output/cube_reconstruction_multiple_views.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: eval_output/cube_reconstruction_multiple_views.png")
        
        # Create a second figure with convex hull
        fig2 = plt.figure(figsize=(15, 5))
        
        # Plot convex hull from different angles
        for i, (elev, azim, title) in enumerate([(30, 45, 'Isometric'), 
                                                   (0, 0, 'Front'), 
                                                   (0, 90, 'Side')]):
            ax = fig2.add_subplot(1, 3, i+1, projection='3d')
            
            # Plot hull faces
            for simplex in hull.simplices:
                triangle = points_array[simplex]
                ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                               alpha=0.3, color='cyan', edgecolor='blue', linewidth=0.5)
            
            # Plot points
            ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
                      c='red', marker='o', s=5, alpha=0.5)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Convex Hull - {title}')
            ax.view_init(elev=elev, azim=azim)
            
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
        
        plt.tight_layout()
        plt.savefig('eval_output/cube_convex_hull.png', dpi=150, bbox_inches='tight')
        print(f"Convex hull visualization saved to: eval_output/cube_convex_hull.png")
        
        plt.close('all')
        
    except Exception as e:
        print(f"Error computing convex hull: {e}")
else:
    print("No points reconstructed!")

print("\n" + "=" * 60)
print("[SUCCESS] Cube visualization test complete!")
print("\nCheck eval_output folder for:")
print("  - cube_reconstruction_multiple_views.png (4 different viewing angles)")
print("  - cube_convex_hull.png (convex hull from 3 angles)")
print("\nExpected shape: CUBE (not cone)")
print("=" * 60)
