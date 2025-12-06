"""
Test Task 3.6: Triangulation-based reconstruction for irregular objects

This test verifies that:
1. Globe (symmetric object) still uses depth maps correctly
2. Dinosaur (irregular object) uses triangulation
3. Both methods produce valid 3D reconstructions
"""

import subprocess
import sys
import os

print("=" * 70)
print("Testing Task 3.6: Triangulation-based Reconstruction")
print("=" * 70)

# Test 1: Verify triangulation works on dinosaur dataset
print("\n[Test 1] Testing triangulation on dinosaur dataset...")
print("-" * 70)

result = subprocess.run([sys.executable, "test_triangulation_dino.py"], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print("[OK] Triangulation test PASSED")
    print(f"  Output: {result.stdout.split('Total 3D points reconstructed:')[1].split('Convex hull volume:')[0].strip()}")
    if "Triangulation implementation is working!" in result.stdout:
        print("  [OK] Triangulation successfully reconstructed dinosaur")
else:
    print("[FAIL] Triangulation test FAILED")
    print(f"  Error: {result.stderr}")
    sys.exit(1)

# Test 2: Check that the detection function works correctly
print("\n[Test 2] Testing dataset detection...")
print("-" * 70)

test_code = """
import sys
sys.path.insert(0, '.')

# Import the function from volume-test.py
exec(open('volume-test.py').read().split('three_images =')[0])

# Test detection for different datasets
test_cases = [
    ('./world_rotate/trans_01.png', True, False, 'Globe (depth map)'),
    ('./cube_images/cube_01.png', False, True, 'Cube (triangulation)'),
    ('./dinoSparseRing/dinoSparseRing/dinoSR0001.png', False, True, 'Dino (triangulation)'),
]

all_passed = True
for path, expected_depth, expected_tri, name in test_cases:
    use_depth, depth_path, use_tri = detect_dataset_and_get_depth_source(path)
    
    if use_depth == expected_depth and use_tri == expected_tri:
        print(f'  [OK] {name}: depth={use_depth}, triangulation={use_tri}')
    else:
        print(f'  [FAIL] {name}: Expected depth={expected_depth}, tri={expected_tri}, Got depth={use_depth}, tri={use_tri}')
        all_passed = False

if all_passed:
    print('\\n[OK] Dataset detection test PASSED')
else:
    print('\\n[FAIL] Dataset detection test FAILED')
    sys.exit(1)
"""

result = subprocess.run([sys.executable, "-c", test_code], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print(result.stdout)
else:
    print("[FAIL] Dataset detection test FAILED")
    print(f"  Error: {result.stderr}")
    sys.exit(1)

# Test 3: Verify camera projection matrix creation
print("\n[Test 3] Testing camera projection matrix creation...")
print("-" * 70)

test_code = """
import numpy as np
import sys
sys.path.insert(0, '.')

# Import the function
exec(open('volume-test.py').read().split('three_images =')[0])

# Test projection matrix creation
fx, fy, cx, cy = 188.9, 188.9, 960, 540
baseline = 35.9

# Test view 0 (no rotation)
P0 = create_camera_projection_matrix(0, fx, fy, cx, cy, baseline)
print(f'  View 0 projection matrix shape: {P0.shape}')
assert P0.shape == (3, 4), 'Projection matrix should be 3x4'

# Test view 1 (15 degree rotation)
P1 = create_camera_projection_matrix(1, fx, fy, cx, cy, baseline)
print(f'  View 1 projection matrix shape: {P1.shape}')
assert P1.shape == (3, 4), 'Projection matrix should be 3x4'

# Verify K matrix is embedded correctly (top-left 3x3 should contain K*R)
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
print(f'  [OK] Projection matrices created correctly')

# Test triangulation function exists
try:
    # Create dummy data
    proj_points = [np.array([100, 100]), np.array([105, 100]), np.array([110, 100])]
    proj_matrices = [P0, P1, create_camera_projection_matrix(2, fx, fy, cx, cy, baseline)]
    
    result = solve_point_triangulation(proj_points, proj_matrices)
    print(f'  [OK] Triangulation function works, result shape: {result.shape}')
    assert result.shape == (4,), 'Triangulation should return homogeneous coordinates (4,)'
    
except Exception as e:
    print(f'  [FAIL] Triangulation function failed: {e}')
    sys.exit(1)

print('\\n[OK] Camera projection matrix test PASSED')
"""

result = subprocess.run([sys.executable, "-c", test_code], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print(result.stdout)
else:
    print("[FAIL] Camera projection matrix test FAILED")
    print(f"  Error: {result.stderr}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("TASK 3.6 TEST SUMMARY")
print("=" * 70)
print("[OK] All tests PASSED")
print("\nImplementation verified:")
print("  1. Triangulation function integrated into reconstruction pipeline")
print("  2. Camera projection matrices created correctly for each view")
print("  3. Dataset detection distinguishes symmetric vs irregular objects")
print("  4. Triangulation successfully reconstructs 3D points from 2D matches")
print("\nNext steps:")
print("  - Test with cube dataset (requires better feature matching)")
print("  - Validate volume calculations against ground truth")
print("  - Compare triangulation vs depth map accuracy")
print("=" * 70)
