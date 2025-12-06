# Design Document

## Overview

This design addresses critical bugs in the existing volume estimation code that reconstructs 3D objects from multiple 2D images. The system uses multi-view stereo vision to match features across images, triangulate 3D points, and compute object volume.

There are two main implementations to evaluate and fix:
1. **volume-estimation.py**: Uses depth maps and rotation transformations
2. **volume-estimation-PointCloud-ManyImages.py**: Uses SVD-based factorization

Current issues across both implementations include:

1. **Coordinate system confusion**: Mixing image pixel coordinates with world coordinates
2. **Incorrect depth-to-3D conversion**: Using depth map values directly as Z coordinates without proper camera-to-world transformation  
3. **Rotation accumulation errors**: Applying rotations incorrectly when aligning points from different views
4. **Volume calculation inaccuracies**: Using convex hull on improperly reconstructed points

The design will identify which implementation is closer to correct, then specify fixes to address the bugs while preserving the existing code structure.

## Architecture

Both existing implementations follow a similar pipeline architecture with five main stages:

1. **Feature Matching Stage**: Detects and matches SIFT features across image triplets
2. **Stereo Rectification Stage**: Aligns image pairs for disparity computation (volume-estimation.py only)
3. **3D Reconstruction Stage**: Reconstructs 3D points either via:
   - Depth maps + rotation (volume-estimation.py)
   - SVD factorization (volume-estimation-PointCloud-ManyImages.py)
4. **Point Cloud Alignment Stage**: Transforms points to common coordinate system
5. **Volume Computation Stage**: Constructs convex hull and calculates volume (volume-test.py)

### Implementation Comparison

**volume-estimation.py**:
- Uses stereo rectification and disparity computation
- Attempts to use depth maps from Blender
- Applies rotation transformations to align views
- More complex but attempts to use actual depth information

**volume-estimation-PointCloud-ManyImages.py**:
- Uses SVD factorization (Tomasi-Kanade approach)
- Simpler, doesn't require depth maps
- Relies on structure-from-motion principles
- May be more robust for feature-based reconstruction

**Recommendation**: Start with volume-estimation-PointCloud-ManyImages.py as it has fewer dependencies and a cleaner approach, then apply fixes to volume-estimation.py if needed.

## Components and Functions to Fix

### Existing Functions in Both Implementations

**get_point_matches(img1, img2)**:
- Current: Works correctly for feature matching
- Issues: Ratio threshold varies (0.85 vs 0.90), should be consistent
- Fix: Standardize ratio threshold to 0.85

**combine_matches(matches_a, matches_b)**:
- Current: Works correctly to find common features
- Issues: None identified
- Fix: No changes needed

**load_image(filepath)**:
- Current: Loads and normalizes images
- Issues: None identified
- Fix: No changes needed

### Functions Specific to volume-estimation.py

**rectify_two(test1, test2, pts1, pts2, F, path1, path2)**:
- Current: Performs stereo rectification
- Issues: path1, path2 parameters unused
- Fix: Remove unused parameters

**get_depth(img1_rectified, img2_rectified)**:
- Current: Computes disparity map
- Issues: Returns disparity but doesn't convert to depth properly
- Fix: Actually compute and return depth using baseline and focal length

**set_points(three_images_arr, i, depth_map)**:
- Current: Processes image triplet and returns matches
- Issues: Parameter 'i' unused, depth_map handling is incorrect
- Fix: Remove unused parameter, fix depth map coordinate usage

**align_points(X, i)**:
- Current: Attempts to apply rotation
- Issues: Rotation logic is incorrect, applies rotation multiple times incorrectly
- Fix: Apply single rotation based on view index

### Functions Specific to volume-estimation-PointCloud-ManyImages.py

**SVD factorization section**:
- Current: Performs SVD on centered matches to get M and S matrices
- Issues: Doesn't properly handle the affine ambiguity
- Fix: Ensure correct_affine_ambiguity is applied properly

**reproject_image_points(A, X, centers)**:
- Current: Reprojects 3D points back to images
- Issues: Used for validation but not for fixing reconstruction
- Fix: Use reprojection error to validate reconstruction quality

### Functions in volume-test.py

**solve_point_triangulation(proj_points, proj_matrices)**:
- Current: Triangulates 3D point from 2D correspondences
- Issues: Works correctly
- Fix: No changes needed

**Depth-to-3D conversion section**:
- Current: Converts depth map values to 3D coordinates
- Issues: **MAJOR BUG** - Uses depth map values directly as Y coordinate, incorrect formula for X and Z
- Fix: Use proper pinhole camera model: X = (u - cx) * depth / fx, Y = (v - cy) * depth / fy, Z = depth

**Rotation application section**:
- Current: Applies rotation to points from different views
- Issues: Rotation accumulation is incorrect, applies rotation wrong number of times
- Fix: Apply rotation of (view_index * rotation_angle) to each point

**triangleVol(p1, p2, p3)**:
- Current: Computes signed volume of tetrahedron
- Issues: Complex logic for vertex ordering
- Fix: Simplify using standard formula: V = (1/6) * |p1 · (p2 × p3)|

**meshVol(faces)**:
- Current: Sums volumes of all tetrahedra
- Issues: Works correctly
- Fix: No changes needed

## Data Models

### CameraParameters
```python
@dataclass
class CameraParameters:
    K: np.ndarray  # 3x3 intrinsic matrix
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    
    def get_projection_matrix(self) -> np.ndarray:
        """Returns 3x4 projection matrix P = K[R|t]"""
        return self.K @ np.hstack([self.R, self.t])
```

### FeatureMatches
```python
@dataclass
class FeatureMatches:
    points: np.ndarray  # Nx3x2 array: N features, 3 images, (x,y) coords
    colors: np.ndarray  # Nx3 array: RGB colors for visualization
    fundamental_matrices: List[np.ndarray]  # List of 3x3 F matrices
```

### PointCloud
```python
@dataclass
class PointCloud:
    points: np.ndarray  # Nx3 array of 3D points
    colors: Optional[np.ndarray] = None  # Nx3 array of RGB colors
    
    def transform(self, R: np.ndarray, t: np.ndarray) -> 'PointCloud':
        """Apply rigid transformation to all points"""
        transformed = (R @ self.points.T).T + t.T
        return PointCloud(transformed, self.colors)
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Lowe's ratio test filtering

*For any* pair of matched features, the distance ratio between the best match and second-best match should be less than the configured threshold (0.7-0.95).

**Validates: Requirements 1.3**

### Property 2: Epipolar constraint satisfaction

*For any* pair of matched points after RANSAC, the epipolar constraint p2^T * F * p1 should be approximately zero (within numerical tolerance).

**Validates: Requirements 1.4**

### Property 3: Combined matches appear in all views

*For any* feature in the combined match set, that feature should appear in all three input image match sets.

**Validates: Requirements 1.5**

### Property 4: Pixel coordinates are valid

*For any* matched point coordinates, the x and y values should be within the valid pixel range [0, image_width) and [0, image_height) respectively.

**Validates: Requirements 1.6**

### Property 5: Rectified points have same vertical coordinate

*For any* pair of corresponding points after rectification, their y-coordinates should be equal (within a small tolerance for numerical error).

**Validates: Requirements 2.3, 8.2**

### Property 6: Depth formula consistency

*For any* computed depth value with valid disparity (> 0), the relationship depth * disparity = baseline * focal_length should hold.

**Validates: Requirements 3.2**

### Property 7: Image-to-world round trip

*For any* 3D point, projecting it to image coordinates and then triangulating back should produce a point close to the original (within reprojection error tolerance).

**Validates: Requirements 4.2**

### Property 8: Homogeneous coordinate normalization

*For any* triangulated point in homogeneous coordinates, the fourth coordinate (w) should equal 1 after normalization.

**Validates: Requirements 4.4**

### Property 9: Multi-view consistency

*For any* 3D point reconstructed from multiple views, the reprojection error across all views should be below a threshold, indicating consistent coordinate systems.

**Validates: Requirements 4.5**

### Property 10: Rotation matrix properties

*For any* computed rotation matrix R, it should satisfy R^T * R = I (orthogonality) and det(R) = 1 (proper rotation).

**Validates: Requirements 5.5**

### Property 11: Rotation accumulation

*For any* sequence of N rotations by angle θ, the accumulated rotation should equal a single rotation by N*θ.

**Validates: Requirements 5.2**

### Property 12: Rotation axis invariance

*For any* rotation around the z-axis, points on the z-axis (x=0, y=0) should remain unchanged.

**Validates: Requirements 5.3**

### Property 13: Transformation order

*For any* point p transformed by rotation R and translation t, the result should equal R*p + t, not R*(p + t).

**Validates: Requirements 5.4**

### Property 14: Rotation angle verification

*For any* rotation matrix constructed from angle θ, extracting the angle from the matrix should yield θ (within numerical tolerance).

**Validates: Requirements 5.1**

### Property 15: Volume is positive

*For any* computed volume from a valid point cloud, the volume should be a positive number.

**Validates: Requirements 6.4**

### Property 16: Depth is positive

*For any* valid depth value, it should be positive, representing distance from the camera along the optical axis.

**Validates: Requirements 7.2**

### Property 17: Transformation order in camera-to-world

*For any* point in camera coordinates, transforming to world coordinates should apply rotation before translation: p_world = R * p_camera + t.

**Validates: Requirements 7.4**

### Property 18: Matrix dimension compatibility

*For any* chained matrix operations in coordinate transformations, the dimensions should be compatible (e.g., 3x3 @ 3x1 = 3x1).

**Validates: Requirements 7.5**

### Property 19: Reprojection error calculation

*For any* triangulated 3D point, the reprojection error should be computed as the sum of squared distances between original 2D points and reprojected points across all views.

**Validates: Requirements 8.4**

## Error Handling

### Invalid Disparity Values

**Issue**: Disparity can be zero or negative, leading to division by zero or negative depth.

**Solution**:
- Check disparity values before depth conversion
- Set depth to a sentinel value (e.g., infinity or max_depth) for invalid disparities
- Filter out points with invalid depth before triangulation
- Log warning when significant portion of disparities are invalid

### Insufficient Feature Matches

**Issue**: Too few matches between images can lead to poor reconstruction.

**Solution**:
- Require minimum number of matches (e.g., 8 for fundamental matrix, 50 for reliable reconstruction)
- Return error status if insufficient matches found
- Suggest adjusting ratio threshold or using different images

### Degenerate Camera Configurations

**Issue**: Cameras too close together or pointing in nearly same direction produce poor triangulation.

**Solution**:
- Check baseline distance and angle between views
- Compute condition number of triangulation matrix
- Warn if configuration is near-degenerate
- Suggest using views with larger baseline or different angles

### Singular Matrices

**Issue**: Matrix operations can fail if matrices are singular or near-singular.

**Solution**:
- Check matrix rank before inversion
- Use pseudo-inverse for near-singular matrices
- Add regularization term if needed
- Return error status for truly singular cases

### Point Cloud Too Sparse

**Issue**: Too few 3D points lead to inaccurate volume estimation.

**Solution**:
- Check point cloud density
- Require minimum number of points (e.g., 100)
- Compute confidence metric based on point density
- Suggest using more images or different views

## Testing Strategy

### Unit Testing

Unit tests will verify individual components in isolation:

**FeatureMatcher Tests**:
- Test SIFT detection on synthetic images with known features
- Verify ratio test correctly filters matches
- Test RANSAC outlier removal with synthetic outliers
- Verify combine_matches preserves only common features

**StereoRectifier Tests**:
- Test rectification on synthetic stereo pairs
- Verify epipolar lines become horizontal
- Test with various fundamental matrices

**PointTriangulator Tests**:
- Test triangulation with known 3D points and camera parameters
- Verify homogeneous coordinate normalization
- Test with degenerate configurations
- Verify reprojection error calculation

**CoordinateTransformer Tests**:
- Test projection matrix construction
- Verify rotation matrix properties (orthogonality, determinant)
- Test rotation angle extraction
- Verify transformation order (rotation before translation)

**VolumeCalculator Tests**:
- Test volume calculation on simple shapes (cube, sphere)
- Verify volume is always positive
- Test with various point cloud densities

### Property-Based Testing

Property-based tests will verify universal properties across many randomly generated inputs using the Hypothesis library for Python:

**Test Configuration**:
- Each property test will run a minimum of 100 iterations
- Use Hypothesis strategies to generate valid inputs (images, points, matrices, etc.)
- Shrink failing examples to minimal counterexamples

**Property Test Coverage**:
- All 19 correctness properties will be implemented as property-based tests
- Each test will be tagged with the property number and requirement it validates
- Tests will use appropriate generators for each input type (rotation matrices, point clouds, camera parameters, etc.)

### Integration Testing

Integration tests will verify the complete pipeline:

**End-to-End Tests**:
- Test complete pipeline on synthetic data with known ground truth
- Verify volume accuracy within acceptable tolerance
- Test with various numbers of views (3, 6, 12, 24)
- Test with different rotation angles

**Regression Tests**:
- Test on the existing world_rotate and dinoSparseRing datasets
- Compare results before and after fixes
- Verify improvements in volume accuracy

### Test Data

**Synthetic Data**:
- Generate images of simple 3D shapes (cube, sphere, cylinder) with known volumes
- Use Blender or similar tool to render from multiple viewpoints
- Include ground truth camera parameters and 3D points

**Real Data**:
- Use existing world_rotate dataset (cube images)
- Use existing dinoSparseRing dataset (dinosaur model)
- Compare against known volumes if available

## Implementation Notes

### Coordinate System Conventions

The system uses three coordinate systems:

1. **Image Coordinates**: Origin at top-left, x-axis right, y-axis down, units in pixels
2. **Camera Coordinates**: Origin at camera center, z-axis along optical axis (forward), x-axis right, y-axis down
3. **World Coordinates**: Origin at scene center, z-axis up, x-axis right, y-axis forward

Transformations:
- Image to Camera: Apply inverse intrinsics K^-1
- Camera to World: Apply extrinsics [R|t] where p_world = R * p_camera + t
- World to Camera: Apply inverse extrinsics where p_camera = R^T * (p_world - t)

### Camera Parameter Format

Camera parameters are stored as:
- K: 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
- R: 3x3 rotation matrix (camera to world)
- t: 3x1 translation vector (camera position in world coordinates)

Projection matrix: P = K @ [R | t] (3x4 matrix)

### Rotation Accumulation

For turntable rotation with angle θ per step:
- View 0: R_0 = I (identity)
- View 1: R_1 = Rot_z(θ)
- View 2: R_2 = Rot_z(2θ) = Rot_z(θ) @ Rot_z(θ)
- View N: R_N = Rot_z(Nθ)

Important: Rotations are applied in world coordinates, not camera coordinates.

### Volume Calculation Methods

Two methods are implemented:

1. **Convex Hull**: Fast, works for convex objects, uses scipy.spatial.ConvexHull
2. **Signed Tetrahedra**: More accurate for complex shapes, sums signed volumes of tetrahedra formed by mesh faces and origin

For most objects, convex hull provides a good approximation. For concave objects, signed tetrahedra with proper mesh reconstruction is more accurate.

### Performance Considerations

- Feature matching is the slowest step (O(n^2) for n features)
- Use FLANN for faster approximate matching
- Limit number of features if performance is critical (e.g., 1000-2000 per image)
- Triangulation is O(n) for n matched features
- Volume calculation is O(n log n) for convex hull, O(n) for mesh volume

### Known Limitations

- System assumes calibrated or semi-calibrated cameras (known intrinsics)
- Assumes small baseline (< 30 degrees) between consecutive views
- Assumes object is mostly convex for accurate volume estimation
- Requires good texture for feature matching (fails on textureless objects)
- Assumes static scene (no moving objects)

## Dependencies

- OpenCV (cv2): Feature detection, matching, stereo rectification
- NumPy: Matrix operations, array manipulation
- SciPy: Spatial operations (ConvexHull, Delaunay), rotation utilities
- Matplotlib: Visualization
- Hypothesis: Property-based testing
- Pytest: Test framework
