#!/usr/bin/env python
"""
Test to verify that the ratio threshold is correctly set to 0.85 in all implementations.
"""
import re
import os

def test_ratio_threshold_standardization():
    """Verify all get_point_matches functions use 0.85 ratio threshold."""
    
    # Files to check
    files_to_check = [
        'volume-test.py',
        'tests/volume-estimation.py',
        'tests/volume-estimation-PointCloud-ManyImages.py',
        'tests/volume-test-eval.py',
        'tests/volume-test-dino_data.py',
        'tests/volume-estimation-PointCloud-eval.py',
        'tests/test_triangulation_dino.py',
        'tests/test_triangulation_cube.py',
        'tests/test_cube_visualization.py',
        'tests/test_cube_triangulation.py',
        'tests/test_cube_depth_maps.py',
    ]
    
    # Pattern to match ratio test
    pattern = r'if\s+m\.distance\s*<\s*0\.(\d+)\s*\*\s*n\.distance'
    
    results = []
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
            matches = re.findall(pattern, content)
            
            if matches:
                for match in matches:
                    threshold = f"0.{match}"
                    results.append((filepath, threshold))
                    if threshold != "0.85":
                        print(f"FAIL: {filepath} uses threshold {threshold} (expected 0.85)")
                    else:
                        print(f"PASS: {filepath} uses threshold {threshold}")
            else:
                print(f"Warning: No ratio test found in {filepath}")
    
    # Check all are 0.85
    all_correct = all(threshold == "0.85" for _, threshold in results)
    
    if all_correct and results:
        print(f"\n✓ All {len(results)} implementations use the standardized ratio threshold of 0.85")
        return True
    elif not results:
        print("\n✗ No ratio tests found in any files")
        return False
    else:
        print(f"\n✗ Some implementations do not use 0.85 threshold")
        return False

if __name__ == "__main__":
    success = test_ratio_threshold_standardization()
    exit(0 if success else 1)
