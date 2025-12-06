# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install opencv-python numpy matplotlib scipy open3d pyvista scikit-image colorspacious pillow
```

## Running Reconstructions

### Dinosaur (Triangulation)
```bash
# Full reconstruction (all 14 triplets)
python volume-test.py --dataset dino

# Quick test (5 triplets)
python volume-test.py --dataset dino --num-triplets 5
```

**Expected Output:**
- 400-600 triangulated points
- Reprojection error < 1 pixel median
- Bounding box: ~0.15 × 0.08 × 0.10 units
- Volume: ~0.0002 cubic units

### Globe (Depth Maps + Rotation)
```bash
# Full reconstruction (all 12 triplets)
python volume-test.py --dataset world

# Quick test (3 triplets)
python volume-test.py --dataset world --num-triplets 3
```

**Expected Output:**
- 1000-2000 points from feature matching
- Spherical shape with continents visible
- Volume: ~800,000 - 1,400,000 cubic units

### Cube (ICP Alignment)
```bash
# Full reconstruction (all 6 triplets)
python volume-test.py --dataset cube

# Quick test (2 triplets)
python volume-test.py --dataset cube --num-triplets 2
```

**Expected Output:**
- Dense point cloud from depth maps
- Cube-like shape
- Volume varies based on depth map scaling

## Output Files

All visualizations are saved to `output/` directory:
- `{dataset}_3d_scatter.png` - 3D scatter plot of reconstructed points
- `{dataset}_trisurf.png` - Triangulated surface visualization

## Understanding the Output

### Point Cloud Validation
```
Total reconstructed points: 590
Point cloud density is sufficient (590 >= 100)
Bounding box size: X=0.15, Y=0.08, Z=0.10
```

### Volume Calculation
```
Hull volume: 0.00021425975054957115
Calculated volume is: 1.991595195408596e-05
Difference in volume: 0.00019434379859548518
```

**Note:** Use the "Hull volume" as the primary estimate. The "Calculated volume" may differ due to face winding order issues in the Delaunay triangulation.

## Troubleshooting

### Low Point Count
- Try increasing `--num-triplets` to process more images
- Check that images have sufficient texture for feature matching
- For depth maps, ensure proper depth scaling

### High Reprojection Errors
- Check camera calibration parameters
- Verify image quality and focus
- Ensure proper fundamental matrix estimation

### Degenerate Point Cloud
- Increase number of views processed
- Check for sufficient baseline between camera positions
- Verify depth map quality and scaling

## Advanced Usage

### Modify Feature Detection
Edit `volume-test.py` line ~157:
```python
sift = cv.SIFT_create(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=15)
```

### Adjust Sampling Density
Edit `volume-test.py` line ~900:
```python
sample_step = 10  # Sample every N pixels
```

### Change Outlier Filtering
Edit `volume-test.py` line ~780:
```python
error_threshold = median_error + 3 * mad  # Adjust multiplier
```

## Dataset Information

### DinoSparseRing (`test_datasets/dinoSparseRing/`)
- 16 views on a ring around ceramic dinosaur
- Real camera calibration from Stanford spherical gantry
- Bounding box: 0.13 × 0.087 × 0.073 units (from README)

### World Rotate (`test_datasets/world_rotate/`)
- 25 views of rotating globe
- Provided depth map from Blender
- Symmetric object (sphere)

### Cube Images (`test_datasets/cube_images/`)
- 14 depth map views of cube
- Grayscale images where intensity = depth
- Background value: 206 (light gray)

## Performance Tips

1. **Start with fewer triplets** for quick testing
2. **Use `--num-triplets 5`** for development
3. **Process all triplets** for final results
4. **Check output images** to verify reconstruction quality
5. **Monitor reprojection errors** - should be < 5 pixels

## Common Issues

**"Only X matches found (need at least 8)"**
- Normal for low-texture images
- Will fall back to uniform sampling for depth maps

**"Warning: X% of disparity values are invalid"**
- Expected for stereo depth estimation
- Triangulation is preferred for textured objects

**"Point cloud is nearly planar or degenerate"**
- Need more views or better feature distribution
- Try processing more triplets

## Next Steps

1. Review output visualizations in `output/` folder
2. Check FIXES_SUMMARY.md for detailed improvements
3. Read CS_482_Final_Project.pdf for theoretical background
4. Experiment with different datasets and parameters
