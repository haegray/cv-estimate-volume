# Requirements Document

## Introduction

This document specifies the requirements for fixing the volume estimation system that reconstructs 3D objects from multiple 2D images and calculates their volume. The system currently contains errors in coordinate transformations, depth estimation, 3D reconstruction, and volume calculation that result in inaccurate volume estimates.

## Glossary

- **System**: The volume estimation software that processes multiple 2D images to reconstruct 3D objects and calculate volume
- **Feature Matching**: The process of identifying corresponding points across multiple images using SIFT descriptors
- **Stereo Rectification**: The transformation of stereo image pairs to align epipolar lines horizontally
- **Disparity Map**: A 2D map showing the pixel displacement between corresponding points in stereo images
- **Depth Map**: A 2D representation where each pixel value represents the distance from the camera to the object surface
- **Point Cloud**: A set of 3D points representing the surface of an object in world coordinates
- **Convex Hull**: The smallest convex shape that contains all points in a point cloud
- **Camera Intrinsics**: Internal camera parameters including focal length and principal point
- **Camera Extrinsics**: External camera parameters including rotation and translation relative to world coordinates
- **Triangulation**: The process of computing 3D point positions from 2D image correspondences and camera parameters
- **Homogeneous Coordinates**: A coordinate system that uses an extra dimension to represent points (x, y, z, w)

## Requirements

### Requirement 1

**User Story:** As a computer vision researcher, I want accurate feature matching between consecutive images, so that I can reliably track points across multiple views of the rotating object.

#### Acceptance Criteria

1. WHEN the System processes two consecutive images THEN the System SHALL detect keypoints and compute descriptors using SIFT
2. WHEN the System matches SIFT descriptors between images THEN the System SHALL use FLANN-based matcher with KD-tree indexing
3. WHEN the System filters initial matches THEN the System SHALL apply Lowe's ratio test with a threshold between 0.7 and 0.95
4. WHEN the System computes the fundamental matrix THEN the System SHALL use RANSAC-based method to remove outlier matches
5. WHEN the System combines matches from three images THEN the System SHALL preserve only features that appear in all three images
6. WHEN feature matching completes THEN the System SHALL return matched point coordinates in image pixel coordinates

### Requirement 2

**User Story:** As a computer vision researcher, I want correct stereo rectification of image pairs, so that corresponding points lie on the same horizontal scanline for accurate disparity computation.

#### Acceptance Criteria

1. WHEN the System receives two images and their fundamental matrix THEN the System SHALL compute rectification homographies for both images
2. WHEN the System applies rectification THEN the System SHALL warp both images using the computed homographies
3. WHEN rectification is complete THEN the System SHALL ensure corresponding points have the same vertical coordinate
4. WHEN the System processes rectified images THEN the System SHALL preserve the geometric relationships between matched features

### Requirement 3

**User Story:** As a computer vision researcher, I want accurate depth estimation from stereo image pairs, so that I can reconstruct the 3D structure of the object.

#### Acceptance Criteria

1. WHEN the System computes disparity THEN the System SHALL use StereoSGBM algorithm with appropriate block size and disparity range parameters
2. WHEN the System converts disparity to depth THEN the System SHALL apply the formula depth = (baseline * focal_length) / disparity
3. WHEN disparity is zero or negative THEN the System SHALL handle the invalid depth value appropriately
4. WHEN the System computes depth maps THEN the System SHALL normalize depth values to a consistent scale across all image pairs

### Requirement 4

**User Story:** As a computer vision researcher, I want correct 3D point triangulation from 2D image correspondences, so that I can accurately reconstruct the object's geometry in world coordinates.

#### Acceptance Criteria

1. WHEN the System triangulates points THEN the System SHALL use camera projection matrices with correct intrinsic and extrinsic parameters
2. WHEN the System converts image coordinates to world coordinates THEN the System SHALL apply the correct transformation from pixel coordinates to camera coordinates
3. WHEN the System computes 3D positions THEN the System SHALL solve the triangulation using SVD on the linear system
4. WHEN triangulation produces homogeneous coordinates THEN the System SHALL normalize by dividing by the fourth coordinate
5. WHEN the System processes multiple views THEN the System SHALL maintain consistent coordinate system orientation across all views

### Requirement 5

**User Story:** As a computer vision researcher, I want correct rotation transformations applied to reconstructed points, so that points from different camera views are aligned in a common world coordinate system.

#### Acceptance Criteria

1. WHEN the System rotates points between views THEN the System SHALL apply rotation matrices corresponding to the known camera rotation angles
2. WHEN the System processes images with 15-degree rotation increments THEN the System SHALL accumulate rotations correctly for each subsequent view
3. WHEN the System applies rotation THEN the System SHALL rotate around the correct axis (z-axis for turntable rotation)
4. WHEN the System transforms points THEN the System SHALL apply rotation before translation in the transformation pipeline
5. WHEN rotation matrices are computed THEN the System SHALL ensure they are proper rotation matrices (orthogonal with determinant 1)

### Requirement 6

**User Story:** As a computer vision researcher, I want accurate volume calculation from the reconstructed 3D point cloud, so that I can measure the object's volume with minimal error.

#### Acceptance Criteria

1. WHEN the System computes volume from a point cloud THEN the System SHALL construct a convex hull or mesh representation
2. WHEN the System calculates mesh volume THEN the System SHALL use the signed volume of tetrahedra method
3. WHEN the System sums tetrahedron volumes THEN the System SHALL ensure consistent vertex ordering for correct sign
4. WHEN the System computes the final volume THEN the System SHALL return the absolute value to ensure positive volume
5. WHEN point cloud density is insufficient THEN the System SHALL report uncertainty or request additional views

### Requirement 7

**User Story:** As a computer vision researcher, I want proper coordinate system handling throughout the pipeline, so that transformations between image, camera, and world coordinates are mathematically correct.

#### Acceptance Criteria

1. WHEN the System converts between coordinate systems THEN the System SHALL document which coordinate system each variable represents
2. WHEN the System uses depth values THEN the System SHALL consistently interpret depth as distance along the camera's optical axis
3. WHEN the System applies camera intrinsics THEN the System SHALL use the correct focal length and principal point values
4. WHEN the System transforms from camera to world coordinates THEN the System SHALL apply extrinsic parameters in the correct order
5. WHEN coordinate transformations are chained THEN the System SHALL verify dimensional consistency at each step

### Requirement 8

**User Story:** As a computer vision researcher, I want validation of intermediate results, so that I can identify where errors occur in the reconstruction pipeline.

#### Acceptance Criteria

1. WHEN the System completes feature matching THEN the System SHALL report the number of matched features
2. WHEN the System computes rectification THEN the System SHALL verify that epipolar lines are horizontal
3. WHEN the System generates a disparity map THEN the System SHALL visualize the disparity for quality assessment
4. WHEN the System triangulates points THEN the System SHALL compute and report reprojection error
5. WHEN the System calculates volume THEN the System SHALL compare against known ground truth if available
