"""
Test cube reconstruction using uniform sampling from depth maps.
This is the proper approach for cube images which ARE depth maps.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull

print("Testing cube dataset with uniform sampling from depth maps...")
print("=" * 60)

# Test with cube images
three_images_coords = ['./cube_images/cube_01.png', './cube_images/cube_02.png', './cube_images/cube_03.png']

# Load images (they are depth maps)
test1 = cv.imread(three_images_coords[0], 0)
test2 = cv.imread(three_images_coords[1], 0)
test3 = cv.imread(three_images_coords[2], 0)

print(f"Image shapes: {test1.shape}")
print(f"Image 1 range: {test1.min()}-{test1.max()}, mean: {test1.mean():.1f}")

# Use the images themselves as depth maps
# Cube images: darker = closer, so invert
depth_map_1 = 255 - test1
depth_map_2 = 255 - test2
depth_map_3 = 255 - test3

depth_maps = [depth_map_1, depth_map_2, depth_map_3]

print(f"Inverted depth maps (darker = closer in original)")
print(f"Depth map 1 range: {depth_map_1.min()}-{depth_map_1.max()}")

# Camera parameters
height, width = test1.shape
fx = 188.9763779528
fy = 188.9763779528
cx = width / 2.0
cy = height / 2.0

# Sample points uniformly from each depth map
sample_step = 20  # Sample every 20 pixels
all_points = []

for view_idx, depth_map_for_view in enumerate(depth_maps):
    view_points = []
    
    # Sample uniformly across the image
    for v in range(0, height, sample_step):
        for u in range(0, width, sample_step):
            # Bounds checking
            if v >= height or u >= width:
                continue
            
            # Get depth value
            depth = depth_map_for_view[v, u]
            
            # Skip invalid/background points (very low depth = background)
            if depth < 10:  # Threshold for background
                continue
            
            # Apply pinhole camera model
            X = (u - cx) * depth / fx
            Y = (v - cy) * depth / fy
            Z = depth
            
            view_points.append((X, Y, Z))
    
    print(f"View {view_idx + 1}: Sampled {len(view_points)} points")
    
    # Apply rotation to transform from camera to world coordinates
    # View 0: 0 degrees, View 1: 15 degrees, View 2: 30 degrees
    view_index = view_idx
    camera_angle = 15 * view_index
    rotation_radians = np.radians(camera_angle)
    rotation_axis = np.array([0, 0, 1])
    rotation_vector = rotation_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    
    # Apply rotation to all points
    if len(view_points) > 0:
        view_points_array = np.array(view_points)
        rotated_points = rotation.apply(view_points_array)
        all_points.extend(rotated_points.tolist())

print(f"\n{'=' * 60}")
print(f"Total 3D points reconstructed: {len(all_points)}")

if len(all_points) > 4:
    # Convert to numpy array
    points_array = np.array(all_points)
    
    print(f"Point cloud bounds:")
    print(f"  X: {points_array[:, 0].min():.2f} to {points_array[:, 0].max():.2f}")
    print(f"  Y: {points_array[:, 1].min():.2f} to {points_array[:, 1].max():.2f}")
    print(f"  Z: {points_array[:, 2].min():.2f} to {points_array[:, 2].max():.2f}")
    
    # Compute convex hull and volume
    try:
        hull = ConvexHull(points_array)
        print(f"\nConvex hull volume: {hull.volume:.2f}")
        print(f"Number of hull vertices: {len(hull.vertices)}")
        print(f"Number of hull faces: {len(hull.simplices)}")
        
        # Visualize
        fig = plt.figure(figsize=(12, 5))
        
        # 3D scatter plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
                   c=points_array[:, 2], cmap='viridis', marker='o', s=1, alpha=0.5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Cube Reconstruction (Uniform Sampling)')
        
        # Set equal aspect ratio
        max_range = np.array([points_array[:, 0].max() - points_array[:, 0].min(),
                              points_array[:, 1].max() - points_array[:, 1].min(),
                              points_array[:, 2].max() - points_array[:, 2].min()]).max() / 2.0
        
        mid_x = (points_array[:, 0].max() + points_array[:, 0].min()) * 0.5
        mid_y = (points_array[:, 1].max() + points_array[:, 1].min()) * 0.5
        mid_z = (points_array[:, 2].max() + points_array[:, 2].min()) * 0.5
        
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Convex hull visualization
        ax2 = fig.add_subplot(122, projection='3d')
        for simplex in hull.simplices:
            triangle = points_array[simplex]
            ax2.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                            alpha=0.3, color='cyan', edgecolor='black', linewidth=0.1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Convex Hull')
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig('cube_uniform_sampling_test.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: cube_uniform_sampling_test.png")
        print("\n[SUCCESS] Uniform sampling reconstruction is working!")
        
    except Exception as e:
        print(f"Error computing convex hull: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nNot enough points reconstructed!")

print("\nTest complete!")
