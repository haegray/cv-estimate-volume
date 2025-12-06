"""
Debug cube depth map values to understand the reconstruction issue.
"""

import cv2 as cv
import numpy as np

# Load a cube image
img = cv.imread('./cube_images/cube_01.png', 0)

print("Cube image analysis:")
print(f"Shape: {img.shape}")
print(f"Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.1f}")
print(f"Std dev: {img.std():.1f}")

# Invert (darker = closer)
depth = 255 - img

print(f"\nAfter inversion (depth):")
print(f"Min: {depth.min()}, Max: {depth.max()}, Mean: {depth.mean():.1f}")
print(f"Std dev: {depth.std():.1f}")

# Check the depth range
depth_range = depth.max() - depth.min()
print(f"\nDepth range: {depth_range}")

# Sample some points
height, width = img.shape
fx = 188.9763779528
fy = 188.9763779528
cx = width / 2.0
cy = height / 2.0

# Sample a few points
sample_points = [
    (width//4, height//4),
    (width//2, height//2),
    (3*width//4, 3*height//4),
]

print(f"\nSample 3D coordinates (using pinhole model):")
print(f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

for u, v in sample_points:
    d = depth[v, u]
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    print(f"  Pixel ({u}, {v}): depth={d}, 3D=({X:.2f}, {Y:.2f}, {Z:.2f})")

# The issue: depth values are in pixel intensity (0-255), not real depth!
# We need to scale them to actual depth values

print(f"\n[ISSUE IDENTIFIED]")
print(f"Depth values are pixel intensities (0-255), not real world depth!")
print(f"This causes Z range to be ~{depth_range}, while X,Y ranges are much larger")
print(f"due to the (u-cx)*depth/fx formula amplifying the coordinates.")

print(f"\n[SOLUTION]")
print(f"We need to scale depth values to match the expected object size.")
print(f"For a cube, if we assume it's ~100 units wide, depth should also be ~100 units.")
print(f"Current depth range: {depth_range}")
print(f"Suggested scaling factor: ~{100.0 / depth_range:.2f}")
