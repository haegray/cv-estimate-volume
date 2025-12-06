"""
Debug rotation application to understand why points are stacking.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulate what happens in the code
# For a triplet at cmap_option = 0, we have 3 views
# Each view should be rotated by a different angle

print("Simulating rotation for triplet 0:")
print("="*60)

# Create a simple test point
test_point = np.array([100, 0, 50])  # A point on the surface
print(f"Original test point: {test_point}")

# Simulate what the code does
cmap_option = 0  # First triplet

for i in range(3):  # Three views in the triplet
    # This is what the code does:
    view_index = (cmap_option * 2) + i
    camera_angle = 15 * view_index
    angle_rad = np.radians(camera_angle)
    
    rotation = R.from_rotvec(angle_rad * np.array([0, 0, 1]))
    rotated_point = rotation.apply(test_point)
    
    print(f"\nView {i} in triplet {cmap_option}:")
    print(f"  view_index = ({cmap_option} * 2) + {i} = {view_index}")
    print(f"  camera_angle = 15 * {view_index} = {camera_angle}°")
    print(f"  Rotated point: {rotated_point}")

print("\n" + "="*60)
print("PROBLEM IDENTIFIED:")
print("="*60)
print("The formula: view_index = (cmap_option * 2) + i")
print("For triplet 0: view_index = 0, 1, 2 -> angles = 0°, 15°, 30°")
print("For triplet 1: view_index = 2, 3, 4 -> angles = 30°, 45°, 60°")
print("")
print("But the triplets are defined as:")
print("  Triplet 0: trans_01, trans_02, trans_03")
print("  Triplet 1: trans_03, trans_04, trans_05")
print("")
print("trans_01 is at 0°, trans_02 is at 15°, trans_03 is at 30°")
print("So triplet 0 should have angles: 0°, 15°, 30° ✓")
print("")
print("trans_03 is at 30°, trans_04 is at 45°, trans_05 is at 60°")
print("So triplet 1 should have angles: 30°, 45°, 60° ✓")
print("")
print("The formula looks CORRECT!")
print("")
print("BUT WAIT - we're using the SAME depth map for all 3 views!")
print("So we're sampling the same surface 3 times and rotating it.")
print("This creates 3 overlapping copies of the same hemisphere.")
print("")
print("The issue: For a sphere, each camera position sees the SAME hemisphere")
print("(the one facing the camera). So rotating the same depth map gives us")
print("overlapping hemispheres, not a full sphere!")

print("\n" + "="*60)
print("SOLUTION:")
print("="*60)
print("For the world dataset, we should NOT use the same depth map for all views.")
print("Instead, we should:")
print("1. Sample the depth map ONCE per triplet (not 3 times)")
print("2. Apply rotation based on the FIRST image in the triplet")
print("3. OR: Recognize that the depth map already represents the visible surface")
print("   from that camera angle, so we don't need to rotate it at all!")
print("")
print("Actually, the depth map is from a SPECIFIC camera position.")
print("We need to know WHICH camera position it corresponds to.")
print("If it's from trans_01 (0°), then we should only use it for trans_01.")
print("For trans_02 (15°), we'd need a different depth map.")
