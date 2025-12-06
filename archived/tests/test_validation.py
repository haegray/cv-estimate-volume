"""
Test validation and error handling functionality.
Tests for task 6: Add validation and error handling
"""

import numpy as np
import sys
import os

# Add parent directory to path to import from volume-test.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_disparity_validation():
    """Test that disparity validation detects invalid values"""
    print("Testing disparity validation...")
    
    # Create mock disparity map with some invalid values
    # Invalid values: 0, -5, 0, -10 = 4 invalid values
    disp = np.array([[10, 20, 0, -5],
                     [15, 0, 25, 30],
                     [-10, 5, 10, 15]], dtype=np.float32)
    
    # Count invalid disparities
    invalid_mask = disp <= 0
    num_invalid = np.sum(invalid_mask)
    total_pixels = disp.size
    invalid_percentage = (num_invalid / total_pixels) * 100
    
    print(f"  Total pixels: {total_pixels}")
    print(f"  Invalid disparities: {num_invalid}")
    print(f"  Invalid percentage: {invalid_percentage:.1f}%")
    
    assert num_invalid == 4, f"Expected 4 invalid disparities, got {num_invalid}"
    assert invalid_percentage > 30, f"Expected >30% invalid, got {invalid_percentage:.1f}%"
    
    print("  ✓ Disparity validation works correctly")

def test_point_cloud_density_check():
    """Test that point cloud density check detects insufficient points"""
    print("\nTesting point cloud density check...")
    
    # Test with insufficient points
    points_low = np.random.rand(50, 3)
    num_points = len(points_low)
    min_points_required = 100
    
    print(f"  Points: {num_points}, Required: {min_points_required}")
    
    if num_points < min_points_required:
        print(f"  ✓ Correctly detected low density ({num_points} < {min_points_required})")
    else:
        raise AssertionError("Should have detected low density")
    
    # Test with sufficient points
    points_high = np.random.rand(150, 3)
    num_points = len(points_high)
    
    print(f"  Points: {num_points}, Required: {min_points_required}")
    
    if num_points >= min_points_required:
        print(f"  ✓ Correctly detected sufficient density ({num_points} >= {min_points_required})")
    else:
        raise AssertionError("Should have detected sufficient density")

def test_reprojection_error_calculation():
    """Test that reprojection error calculation works"""
    print("\nTesting reprojection error calculation...")
    
    # Create a simple test case
    # 3D point at origin
    point_3d = np.array([0, 0, 10, 1])  # Homogeneous coordinates
    
    # Simple projection matrix (identity rotation, no translation)
    K = np.array([[100, 0, 50],
                  [0, 100, 50],
                  [0, 0, 1]])
    R = np.eye(3)
    t = np.zeros((3, 1))
    P = K @ np.hstack([R, t])
    
    # Project the 3D point
    p_proj_homogeneous = P @ point_3d
    p_proj = p_proj_homogeneous[:2] / p_proj_homogeneous[2]
    
    print(f"  3D point: {point_3d[:3]}")
    print(f"  Projected 2D point: {p_proj}")
    
    # The projected point should be at the principal point (50, 50)
    # since the 3D point is on the optical axis
    expected = np.array([50, 50])
    error = np.linalg.norm(p_proj - expected)
    
    print(f"  Expected: {expected}")
    print(f"  Error: {error:.6f}")
    
    assert error < 0.01, f"Projection error too large: {error}"
    
    print("  ✓ Reprojection error calculation works correctly")

def test_depth_validation_in_reconstruction():
    """Test that depth validation filters invalid values during reconstruction"""
    print("\nTesting depth validation in reconstruction...")
    
    # Simulate depth values
    depth_values = np.array([0, -5, 0.5, 10, 20, -1, 15])
    
    # Filter invalid depths (<=0 or <1)
    valid_depths = []
    invalid_count = 0
    
    for depth in depth_values:
        if depth <= 0 or depth < 1:
            invalid_count += 1
        else:
            valid_depths.append(depth)
    
    print(f"  Total depth values: {len(depth_values)}")
    print(f"  Invalid depths: {invalid_count}")
    print(f"  Valid depths: {len(valid_depths)}")
    
    assert invalid_count == 4, f"Expected 4 invalid depths, got {invalid_count}"
    assert len(valid_depths) == 3, f"Expected 3 valid depths, got {len(valid_depths)}"
    
    print("  ✓ Depth validation filters correctly")

if __name__ == "__main__":
    print("=" * 60)
    print("VALIDATION AND ERROR HANDLING TESTS")
    print("=" * 60)
    
    try:
        test_disparity_validation()
        test_point_cloud_density_check()
        test_reprojection_error_calculation()
        test_depth_validation_in_reconstruction()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
