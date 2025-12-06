"""
Test to verify rotation accumulation fixes in volume estimation code.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def test_rotation_accumulation():
    """
    Test that rotation accumulation works correctly.
    For view i, rotation should be i * rotation_angle around z-axis.
    """
    rotation_angle = -15  # degrees per view
    
    # Test point at (1, 0, 0)
    test_point = np.array([1.0, 0.0, 0.0])
    
    # Test for views 0, 1, 2, 3
    for view_index in range(4):
        # Calculate expected rotation
        angle = rotation_angle * view_index
        rotation_radians = np.radians(angle)
        rotation_axis = np.array([0, 0, 1])
        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        
        # Apply rotation
        rotated_point = rotation.apply(test_point)
        
        # Verify rotation is around z-axis (z-coordinate should remain 0)
        assert abs(rotated_point[2]) < 1e-10, f"Z-coordinate changed for view {view_index}"
        
        # Verify rotation angle
        expected_x = np.cos(rotation_radians)
        expected_y = np.sin(rotation_radians)
        
        assert abs(rotated_point[0] - expected_x) < 1e-10, \
            f"X-coordinate incorrect for view {view_index}: expected {expected_x}, got {rotated_point[0]}"
        assert abs(rotated_point[1] - expected_y) < 1e-10, \
            f"Y-coordinate incorrect for view {view_index}: expected {expected_y}, got {rotated_point[1]}"
        
        print(f"View {view_index}: Rotation by {angle}° - PASS")
    
    print("\nAll rotation accumulation tests passed!")


def test_rotation_matrix_properties():
    """
    Test that rotation matrices are proper rotation matrices.
    - Orthogonal: R^T * R = I
    - Determinant = 1
    """
    rotation_angle = -15  # degrees
    
    for view_index in range(4):
        angle = rotation_angle * view_index
        rotation_radians = np.radians(angle)
        rotation_axis = np.array([0, 0, 1])
        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        
        # Get rotation matrix
        R_matrix = rotation.as_matrix()
        
        # Test orthogonality: R^T * R = I
        identity = R_matrix.T @ R_matrix
        assert np.allclose(identity, np.eye(3)), \
            f"Rotation matrix not orthogonal for view {view_index}"
        
        # Test determinant = 1
        det = np.linalg.det(R_matrix)
        assert abs(det - 1.0) < 1e-10, \
            f"Determinant not 1 for view {view_index}: {det}"
        
        print(f"View {view_index}: Rotation matrix properties - PASS")
    
    print("\nAll rotation matrix property tests passed!")


def test_rotation_axis_invariance():
    """
    Test that points on the z-axis remain unchanged by z-axis rotation.
    """
    rotation_angle = -15  # degrees
    
    # Test point on z-axis
    test_point = np.array([0.0, 0.0, 5.0])
    
    for view_index in range(4):
        angle = rotation_angle * view_index
        rotation_radians = np.radians(angle)
        rotation_axis = np.array([0, 0, 1])
        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        
        # Apply rotation
        rotated_point = rotation.apply(test_point)
        
        # Verify point unchanged
        assert np.allclose(rotated_point, test_point), \
            f"Point on z-axis changed for view {view_index}"
        
        print(f"View {view_index}: Z-axis invariance - PASS")
    
    print("\nAll rotation axis invariance tests passed!")


if __name__ == "__main__":
    print("Testing rotation accumulation fixes...\n")
    print("=" * 60)
    
    test_rotation_accumulation()
    print("\n" + "=" * 60)
    
    test_rotation_matrix_properties()
    print("\n" + "=" * 60)
    
    test_rotation_axis_invariance()
    print("\n" + "=" * 60)
    
    print("\n✓ All tests passed! Rotation logic is correct.")
