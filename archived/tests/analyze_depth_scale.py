"""
Analyze the depth map scaling issue.
"""
import cv2 as cv
import numpy as np

# Load depth map
depth_map = cv.imread('./world_rotate/depth_map_world.png')
depth_map = cv.cvtColor(depth_map, cv.COLOR_BGR2GRAY)

print("Depth Map Analysis:")
print(f"  Shape: {depth_map.shape}")
print(f"  Min: {depth_map.min()}")
print(f"  Max: {depth_map.max()}")
print(f"  Mean: {depth_map.mean():.2f}")
print(f"  Std: {depth_map.std():.2f}")

# The depth values are in pixel intensities (0-255)
# But they represent actual distances in Blender units
# We need to understand the scale

# Camera parameters
fx = fy = 188.9763779528
cx = 960.0
cy = 540.0
b = 35.9

print(f"\nCamera parameters:")
print(f"  Focal length: {fx:.2f}")
print(f"  Principal point: ({cx:.2f}, {cy:.2f})")
print(f"  Baseline: {b:.2f}")

# For a sphere, if the camera is at distance ~baseline from the center,
# and the sphere has radius R, then:
# - Points on the sphere surface should have depth ≈ baseline - R to baseline + R
# - The depth variation should be ≈ 2*R

# From the depth map:
depth_variation = depth_map.max() - depth_map.min()
print(f"\nDepth variation: {depth_variation}")

# The X, Y coordinates are computed as:
# X = (u - cx) * depth / fx
# For u at the edge (u = 1920), depth = 200:
# X = (1920 - 960) * 200 / 188.98 = 960 * 200 / 188.98 ≈ 1015

# So the X range is about ±1015
# But the Z range is only 11 to 206, a range of 195

# This means the depth map values are NOT in the same units as the world coordinates!
# The depth map is probably normalized to 0-255 for visualization

# Let's figure out the correct scale
# If the camera is at distance ~baseline = 35.9 from origin,
# and the sphere has radius ~10 (guessing),
# then depth should range from ~25.9 to ~45.9

# But we're getting depth values of 11 to 206
# This suggests the depth map is scaled incorrectly

print(f"\nPROBLEM IDENTIFIED:")
print(f"  The depth map values (11-206) are in pixel intensities,")
print(f"  but they should be in world units (same as baseline = {b:.2f})")
print(f"  ")
print(f"  The depth map needs to be rescaled to match the world coordinate system.")
print(f"  ")
print(f"  Current Z range: {depth_map.min()} to {depth_map.max()}")
print(f"  Expected Z range for a sphere at distance ~{b:.2f}: ~{b-15:.2f} to ~{b+15:.2f}")
print(f"  ")
print(f"  Scale factor needed: ~{b / depth_map.mean():.3f}")

# Let's check what the correct scale should be
# If the sphere is centered at origin with radius R,
# and camera is at distance baseline from origin,
# then depth to sphere surface ranges from (baseline - R) to (baseline + R)

# From the X, Y ranges, we can estimate the sphere radius
# X range is about ±1015, which is way too large for a sphere at distance 35.9
# This suggests the depth values are being used incorrectly

# Let's recalculate assuming depth should be scaled
scale_factor = b / depth_map.mean()
scaled_depth_min = depth_map.min() * scale_factor
scaled_depth_max = depth_map.max() * scale_factor

print(f"\nIf we scale depth by {scale_factor:.3f}:")
print(f"  Scaled Z range: {scaled_depth_min:.2f} to {scaled_depth_max:.2f}")

# Now let's see what X range we'd get
u_edge = 1920
depth_scaled = depth_map.mean() * scale_factor
X_edge = (u_edge - cx) * depth_scaled / fx
print(f"  X at edge would be: ±{X_edge:.2f}")

print(f"\nCONCLUSION:")
print(f"  The depth map values need to be scaled by ~{scale_factor:.3f}")
print(f"  OR the depth map is already correct and we need to adjust camera parameters")
print(f"  OR the depth map represents something different (e.g., normalized depth)")
