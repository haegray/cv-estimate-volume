"""
Test to diagnose world dataset reconstruction issues.
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Test with just one triplet
three_images_coords = ['./world_rotate/trans_01.png','./world_rotate/trans_02.png','./world_rotate/trans_03.png']

# Load images
test1 = cv.imread(three_images_coords[0], 0)
test2 = cv.imread(three_images_coords[1], 0)
test3 = cv.imread(three_images_coords[2], 0)

print(f"Image shapes: {test1.shape}, {test2.shape}, {test3.shape}")

# Load depth map
depth_map_path = './world_rotate/depth_map_world.png'
depth_map = cv.imread(depth_map_path)
depth_map = cv.cvtColor(depth_map, cv.COLOR_BGR2GRAY)

height, width = test1.shape
depth_map = cv.resize(depth_map, (width, height), interpolation=cv.INTER_AREA)

print(f"Depth map shape: {depth_map.shape}")
print(f"Depth map range: [{depth_map.min()}, {depth_map.max()}]")
print(f"Depth map mean: {depth_map.mean():.2f}")

# Visualize depth map
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(test1, cmap='gray')
axes[0].set_title('Image 1')
axes[1].imshow(test2, cmap='gray')
axes[1].set_title('Image 2')
axes[2].imshow(test3, cmap='gray')
axes[2].set_title('Image 3')
axes[3].imshow(depth_map, cmap='viridis')
axes[3].set_title('Depth Map')
plt.savefig('world_images_and_depth.png', dpi=150, bbox_inches='tight')
print("Saved world_images_and_depth.png")
plt.close()

# Camera parameters
fx = fy = 188.9763779528
cx = width / 2.0
cy = height / 2.0
b = 35.9

print(f"\nCamera parameters:")
print(f"  fx, fy: {fx:.2f}")
print(f"  cx, cy: {cx:.2f}, {cy:.2f}")
print(f"  baseline: {b:.2f}")

# Sample points uniformly from depth map
print(f"\nSampling points from depth map...")
sample_step = 10  # Sample every 10 pixels

X_s = []
Y_s = []
Z_s = []

for v in range(0, height, sample_step):
    for u in range(0, width, sample_step):
        depth = depth_map[v, u]
        
        # Skip background
        if depth < 10:
            continue
        
        # Apply pinhole camera model
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        
        X_s.append(X)
        Y_s.append(Y)
        Z_s.append(Z)

print(f"Sampled {len(X_s)} points")
print(f"X range: [{min(X_s):.2f}, {max(X_s):.2f}]")
print(f"Y range: [{min(Y_s):.2f}, {max(Y_s):.2f}]")
print(f"Z range: [{min(Z_s):.2f}, {max(Z_s):.2f}]")

# Visualize point cloud
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_s, Y_s, Z_s, c=Z_s, cmap='viridis', s=1, alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud from Single View (No Rotation)')

# Set equal aspect ratio
max_range = max(max(X_s) - min(X_s), max(Y_s) - min(Y_s), max(Z_s) - min(Z_s)) / 2.0
mid_x = (max(X_s) + min(X_s)) * 0.5
mid_y = (max(Y_s) + min(Y_s)) * 0.5
mid_z = (max(Z_s) + min(Z_s)) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.savefig('world_single_view_no_rotation.png', dpi=150, bbox_inches='tight')
print("Saved world_single_view_no_rotation.png")
plt.close()

# Now apply rotation to simulate multiple views
from scipy.spatial.transform import Rotation as R

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'green', 'blue']
for view_idx in range(3):
    angle_deg = view_idx * 15
    angle_rad = np.radians(angle_deg)
    
    # Create rotation
    rotation = R.from_rotvec(angle_rad * np.array([0, 0, 1]))
    
    # Apply rotation to all points
    points = np.column_stack([X_s, Y_s, Z_s])
    rotated_points = rotation.apply(points)
    
    ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], 
               c=colors[view_idx], s=1, alpha=0.6, label=f'View {view_idx} ({angle_deg}°)')
    
    print(f"\nView {view_idx} (angle={angle_deg}°):")
    print(f"  X range: [{rotated_points[:, 0].min():.2f}, {rotated_points[:, 0].max():.2f}]")
    print(f"  Y range: [{rotated_points[:, 1].min():.2f}, {rotated_points[:, 1].max():.2f}]")
    print(f"  Z range: [{rotated_points[:, 2].min():.2f}, {rotated_points[:, 2].max():.2f}]")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud from Multiple Views (With Rotation)')
ax.legend()

# Set equal aspect ratio
all_points = []
for view_idx in range(3):
    angle_rad = np.radians(view_idx * 15)
    rotation = R.from_rotvec(angle_rad * np.array([0, 0, 1]))
    points = np.column_stack([X_s, Y_s, Z_s])
    rotated_points = rotation.apply(points)
    all_points.extend(rotated_points)

all_points = np.array(all_points)
max_range = max(all_points[:, 0].max() - all_points[:, 0].min(),
                all_points[:, 1].max() - all_points[:, 1].min(),
                all_points[:, 2].max() - all_points[:, 2].min()) / 2.0
mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.savefig('world_multi_view_with_rotation.png', dpi=150, bbox_inches='tight')
print("Saved world_multi_view_with_rotation.png")
plt.close()

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
print("If the multi-view point cloud forms a ring instead of a sphere,")
print("the issue is that we're only rotating the SAME surface points.")
print("For a sphere, each view should see the SAME surface (it's symmetric),")
print("so rotating the same depth map should give us a ring, not a sphere!")
print("\nThe problem: We need DIFFERENT depth maps for each view,")
print("or we need to use the depth map to reconstruct the full 3D surface,")
print("not just the visible surface from one view.")
