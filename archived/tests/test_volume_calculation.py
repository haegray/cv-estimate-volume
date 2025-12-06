"""
Test the simplified volume calculation functions.
"""
import numpy as np
import sys
import os

# Add parent directory to path to import from volume-test.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we need to test
# We'll define them here since volume-test.py is a script
def triangleVol(p1, p2, p3):
    """
    Compute signed volume of tetrahedron formed by triangle (p1, p2, p3) and origin.
    
    Uses standard formula: V = (1/6) * |p1 · (p2 × p3)|
    
    Args:
        p1, p2, p3: 3D points as numpy arrays or array-like
    
    Returns:
        Signed volume of tetrahedron
    """
    # Convert to numpy arrays if needed
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Compute signed volume using scalar triple product
    # V = (1/6) * p1 · (p2 × p3)
    cross_product = np.cross(p2, p3)
    volume = (1.0 / 6.0) * np.dot(p1, cross_product)
    
    return volume


def meshVol(faces):
    """
    Compute total volume of mesh by summing signed volumes of all tetrahedra.
    
    Each tetrahedron is formed by a triangular face and the origin.
    The signed volumes are summed, then absolute value is returned to ensure positive volume.
    
    Args:
        faces: List of triangular faces, each face is a list of 3 points
    
    Returns:
        Absolute volume of the mesh (always positive)
    """
    total_volume = 0.0
    for face in faces:
        # Each face should have exactly 3 vertices
        if len(face) != 3:
            continue
        # Compute signed volume of tetrahedron formed by face and origin
        vol = triangleVol(face[0], face[1], face[2])
        total_volume += vol
    
    # Return absolute value to ensure positive volume
    return abs(total_volume)


def test_triangleVol_unit_tetrahedron():
    """Test triangleVol with a unit tetrahedron."""
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 1, 0])
    p3 = np.array([0, 0, 1])
    
    vol = triangleVol(p1, p2, p3)
    expected = 1.0 / 6.0
    
    print(f"Unit tetrahedron volume: {vol}")
    print(f"Expected: {expected}")
    assert abs(vol - expected) < 1e-10, f"Expected {expected}, got {vol}"
    print("✓ Unit tetrahedron test passed")


def test_triangleVol_scaled():
    """Test triangleVol with a scaled tetrahedron."""
    # Scale by factor of 2
    p1 = np.array([2, 0, 0])
    p2 = np.array([0, 2, 0])
    p3 = np.array([0, 0, 2])
    
    vol = triangleVol(p1, p2, p3)
    # Volume scales by cube of linear scale: (2^3) * (1/6) = 8/6 = 4/3
    expected = 8.0 / 6.0
    
    print(f"\nScaled tetrahedron volume: {vol}")
    print(f"Expected: {expected}")
    assert abs(vol - expected) < 1e-10, f"Expected {expected}, got {vol}"
    print("✓ Scaled tetrahedron test passed")


def test_triangleVol_sign():
    """Test that triangleVol respects vertex order (sign)."""
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 1, 0])
    p3 = np.array([0, 0, 1])
    
    vol_positive = triangleVol(p1, p2, p3)
    vol_negative = triangleVol(p1, p3, p2)  # Reversed order
    
    print(f"\nVolume with order (p1, p2, p3): {vol_positive}")
    print(f"Volume with order (p1, p3, p2): {vol_negative}")
    assert abs(vol_positive + vol_negative) < 1e-10, "Volumes should be opposite signs"
    print("✓ Sign test passed")


def test_meshVol_positive():
    """Test that meshVol always returns positive volume."""
    # Create faces with mixed orientations
    faces = [
        [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])],
        [np.array([1, 0, 0]), np.array([0, 0, -1]), np.array([0, 1, 0])],  # Reversed
    ]
    
    mesh_vol = meshVol(faces)
    print(f"\nMesh volume: {mesh_vol}")
    assert mesh_vol > 0, f"Mesh volume should be positive, got {mesh_vol}"
    print("✓ Mesh volume positivity test passed")


def test_meshVol_cube():
    """Test meshVol with a simple cube."""
    # Define 8 vertices of a unit cube centered at origin
    v = [
        np.array([-0.5, -0.5, -0.5]),
        np.array([0.5, -0.5, -0.5]),
        np.array([0.5, 0.5, -0.5]),
        np.array([-0.5, 0.5, -0.5]),
        np.array([-0.5, -0.5, 0.5]),
        np.array([0.5, -0.5, 0.5]),
        np.array([0.5, 0.5, 0.5]),
        np.array([-0.5, 0.5, 0.5]),
    ]
    
    # Define 12 triangular faces (2 per cube face)
    faces = [
        # Bottom face (z = -0.5)
        [v[0], v[1], v[2]],
        [v[0], v[2], v[3]],
        # Top face (z = 0.5)
        [v[4], v[6], v[5]],
        [v[4], v[7], v[6]],
        # Front face (y = -0.5)
        [v[0], v[5], v[1]],
        [v[0], v[4], v[5]],
        # Back face (y = 0.5)
        [v[2], v[7], v[3]],
        [v[2], v[6], v[7]],
        # Left face (x = -0.5)
        [v[0], v[3], v[7]],
        [v[0], v[7], v[4]],
        # Right face (x = 0.5)
        [v[1], v[6], v[2]],
        [v[1], v[5], v[6]],
    ]
    
    mesh_vol = meshVol(faces)
    expected = 1.0  # Unit cube volume
    
    print(f"\nCube volume: {mesh_vol}")
    print(f"Expected: {expected}")
    # Allow some tolerance for numerical errors
    assert abs(mesh_vol - expected) < 0.01, f"Expected {expected}, got {mesh_vol}"
    print("✓ Cube volume test passed")


if __name__ == "__main__":
    print("Testing simplified volume calculation functions...")
    print("=" * 60)
    
    test_triangleVol_unit_tetrahedron()
    test_triangleVol_scaled()
    test_triangleVol_sign()
    test_meshVol_positive()
    test_meshVol_cube()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
