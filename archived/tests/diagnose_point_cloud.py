"""
Diagnostic script to visualize point clouds and identify reconstruction issues.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def visualize_point_cloud(points, title="Point Cloud", save_path=None):
    """Visualize a 3D point cloud."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', s=1, alpha=0.6)
        
        # Print statistics
        print(f"\n{title}")
        print(f"  Number of points: {len(points)}")
        print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        print(f"  X std: {points[:, 0].std():.2f}")
        print(f"  Y std: {points[:, 1].std():.2f}")
        print(f"  Z std: {points[:, 2].std():.2f}")
        
        # Set equal aspect ratio
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        print(f"\n{title}: NO POINTS!")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def load_points_from_npy(dataset_name):
    """Load saved point cloud from .npy file."""
    try:
        points = np.load('points_vol_est.npy')
        print(f"\nLoaded points from points_vol_est.npy")
        return points
    except FileNotFoundError:
        print(f"\nNo saved points file found")
        return None


def analyze_depth_to_3d_conversion():
    """Analyze the depth-to-3D conversion to identify issues."""
    print("\n" + "="*60)
    print("ANALYZING DEPTH-TO-3D CONVERSION")
    print("="*60)
    
    # Simulate the conversion with test data
    width, height = 640, 480
    fx = fy = 188.9763779528
    cx = width / 2.0
    cy = height / 2.0
    
    print(f"\nCamera parameters:")
    print(f"  fx, fy: {fx:.2f}")
    print(f"  cx, cy: {cx:.2f}, {cy:.2f}")
    print(f"  Image size: {width} x {height}")
    
    # Test a few sample points
    test_points = [
        (cx, cy, 100),  # Center point
        (cx + 100, cy, 100),  # Right of center
        (cx, cy + 100, 100),  # Below center
        (0, 0, 100),  # Top-left corner
        (width-1, height-1, 100),  # Bottom-right corner
    ]
    
    print(f"\nTest depth-to-3D conversion:")
    print(f"  Formula: X = (u - cx) * depth / fx")
    print(f"           Y = (v - cy) * depth / fy")
    print(f"           Z = depth")
    
    for u, v, depth in test_points:
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        print(f"  Pixel ({u:.0f}, {v:.0f}) at depth {depth} -> 3D ({X:.2f}, {Y:.2f}, {Z:.2f})")


def check_rotation_application():
    """Check how rotations are being applied."""
    print("\n" + "="*60)
    print("CHECKING ROTATION APPLICATION")
    print("="*60)
    
    # Test rotation around z-axis
    angle_deg = 15
    angle_rad = np.radians(angle_deg)
    
    # Create rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    R = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta, cos_theta, 0],
                  [0, 0, 1]])
    
    print(f"\nRotation matrix for {angle_deg}° around z-axis:")
    print(R)
    
    # Test point
    p = np.array([10, 0, 5])
    p_rotated = R @ p
    
    print(f"\nTest point: {p}")
    print(f"After rotation: {p_rotated}")
    print(f"Expected: [{10*cos_theta:.2f}, {10*sin_theta:.2f}, 5.00]")
    
    # Check multiple rotations
    print(f"\nAccumulated rotations:")
    p_current = p.copy()
    for i in range(4):
        angle_i = i * angle_rad
        R_i = np.array([[np.cos(angle_i), -np.sin(angle_i), 0],
                        [np.sin(angle_i), np.cos(angle_i), 0],
                        [0, 0, 1]])
        p_rotated = R_i @ p
        print(f"  View {i} (angle={i*angle_deg}°): {p_rotated}")


if __name__ == "__main__":
    print("POINT CLOUD DIAGNOSTIC TOOL")
    print("="*60)
    
    # Analyze the conversion formulas
    analyze_depth_to_3d_conversion()
    
    # Check rotation application
    check_rotation_application()
    
    # Try to load and visualize existing point clouds
    print("\n" + "="*60)
    print("LOADING SAVED POINT CLOUDS")
    print("="*60)
    
    points = load_points_from_npy("world")
    if points is not None and len(points) > 0:
        visualize_point_cloud(points, "Saved Point Cloud", "diagnostic_point_cloud.png")
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run volume-test.py with --dataset world --num-triplets 1")
    print("2. Check the generated point cloud visualization")
    print("3. Compare X, Y, Z ranges - should be roughly equal for a sphere")
    print("4. If Z range is much smaller, depth conversion is wrong")
    print("5. If points form a ring, rotation application is wrong")
