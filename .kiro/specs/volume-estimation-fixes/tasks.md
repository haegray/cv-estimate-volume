# Implementation Plan

- [x] 1. Evaluate and select primary implementation






  - Analyze both volume-estimation.py and volume-estimation-PointCloud-ManyImages.py
  - Run both implementations and compare outputs
  - Document which approach is more promising
  - Select primary implementation to fix first
  - _Requirements: All requirements (evaluation phase)_

- [x] 2. Fix coordinate system and depth-to-3D conversion in volume-test.py





  - This is the most critical bug affecting volume accuracy
  - _Requirements: 4.2, 7.2, 7.4_

- [ ]* 2.1 Write property test for depth-to-3D conversion
  - **Property 7: Image-to-world round trip**
  - **Validates: Requirements 4.2**

- [x] 2.2 Fix depth-to-3D conversion formula


  - Replace incorrect depth map usage with proper pinhole camera model
  - Use formula: X = (u - cx) * depth / fx, Y = (v - cy) * depth / fy, Z = depth
  - Update the section that converts depth map pixel values to 3D coordinates
  - _Requirements: 4.2, 7.2_

- [ ]* 2.3 Write property test for coordinate transformation order
  - **Property 17: Transformation order in camera-to-world**
  - **Validates: Requirements 7.4**

- [x] 2.4 Verify camera intrinsic parameters are used correctly


  - For dinoSparseRing: Use parameters from dinoSR_par.txt file (already available)
  - For world_rotate: Use existing hardcoded parameters (can be refined later if needed)
  - Ensure K matrix is properly constructed from focal length and principal point
  - _Requirements: 7.3_

- [x] 3. Fix rotation transformations and accumulation



  - Correct how rotations are applied to align points from different views
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 3.1 Write property test for rotation matrix properties
  - **Property 10: Rotation matrix properties**
  - **Validates: Requirements 5.5**

- [ ]* 3.2 Write property test for rotation accumulation
  - **Property 11: Rotation accumulation**
  - **Validates: Requirements 5.2**


- [x] 3.3 Fix rotation accumulation logic


  - Apply rotation of (view_index * rotation_angle) to each point
  - Remove incorrect multiple rotation applications
  - Ensure rotations are around correct axis (z-axis)
  - _Requirements: 5.1, 5.2, 5.3_

- [ ]* 3.4 Write property test for rotation axis invariance
  - **Property 12: Rotation axis invariance**
  - **Validates: Requirements 5.3**

- [ ]* 3.5 Write property test for transformation order
  - **Property 13: Transformation order**
  - **Validates: Requirements 5.4**

- [ ] 3.6 Implement triangulation-based reconstruction for irregular objects












  - Integrate existing solve_point_triangulation() function into main reconstruction loop
  - Set up proper camera projection matrices for each view
  - Use triangulation for irregular objects (cube, dinosaur) instead of depth maps
  - Keep depth map method for symmetric objects (globe/sphere)
  - Handle per-view camera parameters (rotation, translation)
  - Test with cube dataset to verify cube shape (not cone)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  - _Note: Triangulation code exists but is not integrated. See TRIANGULATION_TODO.md for details_

- [x] 4. Standardize feature matching parameters




  - Ensure consistent ratio thresholds across implementations
  - _Requirements: 1.3_

- [ ]* 4.1 Write property test for Lowe's ratio test
  - **Property 1: Lowe's ratio test filtering**
  - **Validates: Requirements 1.3**

- [x] 4.2 Standardize ratio threshold to 0.85


  - Update get_point_matches in both implementations
  - _Requirements: 1.3_

- [ ]* 4.3 Write property test for epipolar constraint
  - **Property 2: Epipolar constraint satisfaction**
  - **Validates: Requirements 1.4**

- [ ]* 4.4 Write property test for combined matches
  - **Property 3: Combined matches appear in all views**
  - **Validates: Requirements 1.5**

- [ ] 5. Fix volume calculation





  - Simplify and correct the volume computation from reconstructed points
  - _Requirements: 6.3, 6.4_

- [ ]* 5.1 Write property test for volume positivity
  - **Property 15: Volume is positive**
  - **Validates: Requirements 6.4**

- [x] 5.2 Simplify triangleVol function


  - Use standard signed volume formula: V = (1/6) * |p1 · (p2 × p3)|
  - Remove complex vertex ordering logic
  - _Requirements: 6.3_

- [x] 5.3 Verify meshVol correctly sums tetrahedra volumes


  - Ensure consistent sign handling
  - Return absolute value for final volume
  - _Requirements: 6.4_

- [x] 6. Add validation and error handling






  - Add checks for invalid inputs and edge cases
  - _Requirements: 3.3, 6.5, 8.4_

- [x] 6.1 Add disparity validation


  - Check for zero or negative disparity before depth conversion
  - Handle invalid depth values appropriately
  - _Requirements: 3.3_

- [x] 6.2 Add point cloud density check


  - Verify minimum number of points before volume calculation
  - Report warning if density is insufficient
  - _Requirements: 6.5_

- [x] 6.3 Add reprojection error calculation


  - Compute reprojection error for triangulated points
  - Report error metrics for quality assessment
  - _Requirements: 8.4_

- [ ]* 6.4 Write property test for reprojection error
  - **Property 19: Reprojection error calculation**
  - **Validates: Requirements 8.4**

- [x] 7. Test with existing datasets









  - Run fixed code on world_rotate and dinoSparseRing datasets


  - Compare volume estimates with ground truth if available
  - _Requirements: 8.5_




- [x] 7.1 Test on world_rotate dataset (cube)









  - Run complete pipeline
  - Compute volume and compare with expected cube volume
  - Visualize reconstructed point cloud
  - _Requirements: All requirements_



- [x] 7.2 Test on dinoSparseRing dataset


  - Run complete pipeline
  - Compute volume


  - Visualize reconstructed point cloud
  - _Requirements: All requirements_

- [ ] 7.3 Document volume accuracy improvements
  - Compare before/after volume estimates
  - Document remaining limitations
  - Suggest further improvements if needed
  - _Requirements: 8.5_

- [ ] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
