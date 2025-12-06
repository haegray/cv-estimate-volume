"""
Test volume estimation on cube dataset using computed stereo depth.
This demonstrates the adaptive depth handling for datasets without provided depth maps.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import sys

# Import functions from volume-test.py
sys.path.insert(0, '.')
from volume_test import (
    load_image, get_point_matches, combine_matches, get_match_colors,
    rectify_two, get_depth, detect_dataset_and_get_depth_source
)

# Test with cube images
three_images = [
    ['./cube_images/cube_01.png', './cube_images/cube_02.png', './cube_images/cube_03.png'],
    ['./cube_images/cube_03.png', './cube_images/cube_04.png', './cube_images/cube_05.png'],
    ['./cube_images/cube_05.png', './cube_images/cube_06.png', './cube_images/cube_07.png'],
]

print("Testing cube dataset with computed stereo depth...")
print("=" * 60)

for i, three_images_coords in enumerate(three_images):
    print(f"\nProcessing triplet {i+1}: {three_images_coords[0]}")
    
    # Detect dataset
    use_provided_depth, depth_map_path = detect_dataset_and_get_depth_source(three_images_coords[0])
    print(f"  Use provided depth: {use_provided_depth}")
    
    # Load images
    test1 = cv.imread(three_images_coords[0], 0)
    test2 = cv.imread(three_images_coords[1], 0)
    test3 = cv.imread(three_images_coords[2], 0)
    
    # Get matches
    matches1, pts1, pts2, F = get_point_matches(test1, test2)
    matches2, pts1_1, pts2_1, F_1 = get_point_matches(test1, test3)
    
    print(f"  Matches 1-2: {len(matches1)}, Matches 1-3: {len(matches2)}")
    
    # Rectify
    img1_rectified, img2_rectified = rectify_two(test1, test2, pts1, pts2, F)
    
    # Compute depth from stereo
    if not use_provided_depth:
        depth_map = get_depth(img1_rectified, img2_rectified)
        
        # Invert for cube (darker = closer in cube images)
        if 'cube_' in three_images_coords[0]:
            depth_map = 255 - depth_map
            print(f"  Inverted depth for cube dataset")
        
        print(f"  Computed depth map shape: {depth_map.shape}")
        print(f"  Depth range: {depth_map.min()} - {depth_map.max()}")

print("\n" + "=" * 60)
print("Cube dataset test complete!")
print("\nKey findings:")
print("- Cube dataset does NOT have provided depth maps")
print("- Depth is computed from stereo disparity")
print("- Depth values are inverted (darker = closer in original images)")
