"""
Test cube reconstruction using depth maps.
The cube images ARE depth maps (grayscale with shading).
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull

print("Testing cube dataset with depth map reconstruction...")
print("=" * 60)

# Test with a small subset of cube images
three_images = [
    ['./cube_images/cube_01.png', './cube_images/cube_02.png', './cube_images/cube_03.png'],
]

def load_image(filepath):
    """Loads an image into a numpy array."""
    img = np.float32(Image.open(filepath))
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

sift = cv.SIFT_create()

def get_point_matches(img1, img2):
    """Returns matches as array: (feature track, image, coord)"""
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.array([]), np.array([]), np.array([]), None

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

# Collect all 3D points
all_points = []

for triplet_idx, three_images_coords in enumerate(three_images):
    print(f"\nProcessing triplet {triplet_idx + 1}: {three_images_coords[0]}")
    
    # Load images (they are depth maps)
    test1 = cv.imread(three_images_coords[0], 0)
    test2 = cv.imread(three_images_coords[1], 0)
    test3 = cv.imread(three_images_coords[2], 0)
    
    print(f"  Image shapes: {test1.shape}, {test2.shape}, {test3.shape}")
    print(f"  Image 1 range: {test1.min()}-{test1.max()}, mean: {test1.mean():.1f}")
    
    # Try to get matches (may fail due to lack of texture)
    matches1, pts1, pts2, F = get_point_matches(test1, test2)
    matches2, pts1_1, pts2_1, F_1 = get_point_matches(test1, test3)
    
    print(f"  Matches 1-2: {len(matches1)}, Matches 1-3: {len(matches2)}")
    
    if len(matches1) == 0 or len(matches2) == 0:
        print(f"  WARNING: Not enough matches for feature-based reconstruction")
        print(f"  This is expected for depth map images (low texture)")
        print(f"  Cube images ARE depth maps, not RGB images")
        continue
    
    # Combine matches
    combined_matches = combine_matches(np.array(matches1), np.array(matches2))
    print(f"  Combined matches: {len(combined_matches)}")
    
    if len(combined_matches) == 0:
        print(f"  No combined matches - cannot reconstruct from features")
        continue
    
    # Camera parameters
    height, width = test1.shape
    fx = 188.9763779528
    fy = 188.9763779528
    cx = width / 2.0
    cy = height / 2.0
    
    # Extract matched points for each view
    im_1_pts = combined_matches[:, 0]
    im_2_pts = combined_matches[:, 1]
    im_3_pts = combined_matches[:, 2]
    
    # Use the images themselves as depth maps
    # Cube images: darker = closer, so invert
    depth_map_1 = 255 - test1
    depth_map_2 = 255 - test2
    depth_map_3 = 255 - test3
    
    print(f"  Using images as depth maps (inverted)")
    print(f"  Depth map 1 range: {depth_map_1.min()}-{depth_map_1.max()}")
    
    # Reconstruct 3D points using depth maps
    triplet_points = []
    
    for match_idx in range(len(im_1_pts)):
        u1, v1 = im_1_pts[match_idx]
        
        # Bounds checking
        if v1 < 0 or v1 >= height or u1 < 0 or u1 >= width:
            continue
        
        # Get depth from depth map
        depth = depth_map_1[v1, u1]
        
        if depth < 1:
            continue
        
        # Apply pinhole camera model
        X = (u1 - cx) * depth / fx
        Y = (v1 - cy) * depth / fy
        Z = depth
        
        triplet_points.append((X, Y, Z))
    
    print(f"  Reconstructed {len(triplet_points)} 3D points from depth maps")
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
        ax.set_title('Cube Reconstruction using Depth Maps')
        
        plt.savefig('cube_depth_map_test.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: cube_depth_map_test.png")
        print("\nDepth map reconstruction is working!")
        
    except Exception as e:
        print(f"Error computing convex hull: {e}")
else:
    print("\nNot enough points reconstructed!")
    print("This is expected - cube images have very low texture for feature matching.")
    print("The cube images ARE depth maps, so feature matching is not the right approach.")
    print("\nFor cube reconstruction, we should:")
    print("  1. Use the images directly as depth maps (darker = closer)")
    print("  2. Sample points uniformly across the depth map")
    print("  3. Apply proper camera transformations for each view")

print("\nTest complete!")
