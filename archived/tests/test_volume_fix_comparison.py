"""
Compare old and new triangleVol implementations to verify they produce similar results.
"""
import numpy as np
from operator import itemgetter

def triangleVol_old(p1, p2, p3):
    """Old implementation with complex vertex ordering logic."""
    pts = [p1,p2,p3]
    p1 = max(pts, key=itemgetter(2))
    new_pts = []
    for i in pts:
        if i[0] != p1[0] or i[1] != p1[1] or i[2] != p1[2]:
            new_pts.append(i)
    pts = new_pts
    N = np.linalg.norm(np.cross((pts[0] - p1), (pts[1] - p1)))
    PV = np.linalg.norm([0,0,0] - p1)
    if(np.dot( PV, N ) > 0.0 ):
        p2 = pts[0]
        p3 = pts[1]
    else:
        p2 = pts[1]
        p3 = pts[0]
    
    v321 = p3[0]*p2[1]*p1[2]
    v231 = p2[0]*p3[1]*p1[2]
    v312 = p3[0]*p1[1]*p2[2]
    v132 = p1[0]*p3[1]*p2[2]
    v213 = p2[0]*p1[1]*p3[2]
    v123 = p1[0]*p2[1]*p3[2]
    return (1.0/6.0)*(-v321 + v231 + v312 - v132 - v213 + v123)


def triangleVol_new(p1, p2, p3):
    """New simplified implementation."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    cross_product = np.cross(p2, p3)
    volume = (1.0 / 6.0) * np.dot(p1, cross_product)
    
    return volume


def meshVol_old(faces):
    """Old implementation."""
    vols = []
    for t in faces:
        vols.append(triangleVol_old(t[0], t[1], t[2]))
    return abs(sum(vols))


def meshVol_new(faces):
    """New implementation."""
    total_volume = 0.0
    for face in faces:
        if len(face) != 3:
            continue
        vol = triangleVol_new(face[0], face[1], face[2])
        total_volume += vol
    return abs(total_volume)


# Test with various triangles
print("Testing triangleVol implementations...")
print("=" * 60)

test_cases = [
    ("Unit tetrahedron", 
     np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
    ("Scaled tetrahedron",
     np.array([2, 0, 0]), np.array([0, 2, 0]), np.array([0, 0, 2])),
    ("Random triangle 1",
     np.array([1.5, 2.3, 3.1]), np.array([0.5, 1.2, 2.8]), np.array([2.1, 0.9, 1.5])),
    ("Random triangle 2",
     np.array([10, 5, 8]), np.array([3, 12, 6]), np.array([7, 4, 15])),
]

for name, p1, p2, p3 in test_cases:
    vol_old = triangleVol_old(p1, p2, p3)
    vol_new = triangleVol_new(p1, p2, p3)
    diff = abs(vol_old - vol_new)
    
    print(f"\n{name}:")
    print(f"  Old: {vol_old:.10f}")
    print(f"  New: {vol_new:.10f}")
    print(f"  Diff: {diff:.10e}")
    
    if diff > 1e-6:
        print(f"  ⚠ WARNING: Significant difference!")

# Test with a mesh
print("\n" + "=" * 60)
print("Testing meshVol with a cube...")

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

faces = [
    [v[0], v[1], v[2]], [v[0], v[2], v[3]],
    [v[4], v[6], v[5]], [v[4], v[7], v[6]],
    [v[0], v[5], v[1]], [v[0], v[4], v[5]],
    [v[2], v[7], v[3]], [v[2], v[6], v[7]],
    [v[0], v[3], v[7]], [v[0], v[7], v[4]],
    [v[1], v[6], v[2]], [v[1], v[5], v[6]],
]

mesh_vol_old = meshVol_old(faces)
mesh_vol_new = meshVol_new(faces)
diff = abs(mesh_vol_old - mesh_vol_new)

print(f"\nCube mesh volume:")
print(f"  Old: {mesh_vol_old:.10f}")
print(f"  New: {mesh_vol_new:.10f}")
print(f"  Diff: {diff:.10e}")
print(f"  Expected: 1.0")

if diff > 1e-6:
    print(f"  ⚠ WARNING: Significant difference!")
else:
    print(f"  ✓ Results match!")
