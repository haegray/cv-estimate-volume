# 3D Volume Estimation from Multiple Images

Computer vision project for estimating the volume of 3D objects from multiple 2D images using stereo reconstruction and triangulation techniques.

## Overview

This project implements multiple approaches for 3D reconstruction and volume estimation:
- **Stereo depth estimation** for textured objects
- **Triangulation with camera calibration** for accurate 3D point reconstruction
- **Depth map integration** for objects with known depth information
- **Known rotation alignment** for multi-view depth map registration

## Features

- SIFT and ORB feature detection with CLAHE preprocessing
- Robust RANSAC-based fundamental matrix estimation
- Adaptive outlier filtering using reprojection error
- Support for multiple datasets (globe, cube, dinosaur)
- Convex hull volume calculation with validation

## Installation

### Requirements
- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- SciPy
- Open3D
- PyVista
- scikit-image
- colorspacious

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy matplotlib scipy open3d pyvista scikit-image colorspacious pillow
```

## Usage

### Command Line Interface

Run the main reconstruction script with different datasets:

```bash
# Reconstruct dinosaur (triangulation with real camera parameters)
python volume-test.py --dataset dino

# Reconstruct globe (depth map with rotation)
python volume-test.py --dataset world

# Reconstruct cube (90-degree rotation alignment of depth maps)
python volume-test.py --dataset cube

# Process limited number of image triplets
python volume-test.py --dataset dino --num-triplets 5

# Enable intermediate visualizations (feature matches, depth maps, rectified images)
python volume-test.py --dataset dino --visualize
```

### Datasets

**1. DinoSparseRing** (`dinoSparseRing/`)
- 16 views of a ceramic dinosaur on a ring
- Real camera calibration parameters
- Uses triangulation for accurate 3D reconstruction
- Expected volume: ~0.0002 cubic units

**2. World Rotate** (`world_rotate/`)
- Globe images with provided depth maps
- Camera rotates around stationary object
- Uses depth map with rotation transformation

**3. Cube Images** (`cube_images/`)
- Depth maps from different viewing angles (corner views)
- Uses 90-degree rotation alignment around Z-axis
- Expected volume: ~22,788 cubic units (84% of theoretical cube)

## Project Structure

```
cv-estimate-volume/
├── volume-test.py              # Main reconstruction script
├── reconstruction_methods.py   # Additional MVS reconstruction methods
├── README.md                   # Project documentation
├── QUICK_START.md              # Quick start guide
├── FIXES_SUMMARY.md            # Improvements log
├── CS_482_Final_Project.pdf    # Full research paper
│
├── test_datasets/              # All test datasets
│   ├── dinoSparseRing/         # Dinosaur dataset
│   │   ├── dinoSR*.png         # 16 images
│   │   ├── dinoSR_par.txt      # Camera parameters
│   │   └── README.txt          # Dataset documentation
│   │
│   ├── world_rotate/           # Globe dataset
│   │   ├── trans_*.png         # 25 images
│   │   └── depthmap.png        # Provided depth map
│   │
│   └── cube_images/            # Cube depth maps
│       └── cube_*.png          # 14 depth map images
│
├── output/                     # Generated visualizations (created on first run)
│   ├── {dataset}_3d_scatter.png      # 3D point cloud scatter plot
│   ├── {dataset}_trisurf.png         # Triangulated surface mesh
│   ├── {dataset}_matches_*.png       # Feature matches (with --visualize)
│   ├── {dataset}_depth_*.png         # Depth maps (with --visualize)
│   └── {dataset}_rectified_*.png     # Rectified stereo pairs (with --visualize)
│
└── archived/                   # Previous test results and experiments
    ├── tests/                  # Old test scripts
    ├── output/                 # Previous output files
    └── eval_output/            # Previous evaluation results
```

## Algorithm Details

### Feature Matching Improvements
- **SIFT with relaxed thresholds**: 5000 features, contrast threshold 0.03
- **CLAHE preprocessing**: Adaptive histogram equalization for better contrast
- **Lowe's ratio test**: 0.8 threshold for more permissive matching
- **RANSAC**: 3.0 pixel threshold with 99% confidence

### Triangulation
- Uses real camera parameters (K, R, t) from calibration
- Solves for 3D points using SVD on projection equations
- Adaptive outlier filtering: median + 3×MAD reprojection error
- Processes all pairwise image combinations for dense coverage

### Volume Calculation
1. Reconstruct 3D point cloud from multiple views
2. Compute Delaunay triangulation
3. Calculate convex hull volume
4. Validate with signed tetrahedron volumes

## Results

### Dinosaur Reconstruction
- **Points**: 590 triangulated points (from 14 triplets)
- **Reprojection error**: < 1 pixel median
- **Bounding box**: 0.15 × 0.08 × 0.10 units
- **Volume**: ~0.0002 cubic units

### Key Improvements
- Increased feature matches from 23 to 590 points
- Reduced reprojection error from 970px to <1px median
- Correct geometric scale matching ground truth

## Validation

The reconstruction includes several validation checks:
- Minimum point count (100 points recommended)
- Bounding box dimension analysis
- Reprojection error statistics
- Degenerate point cloud detection

## Documentation

- **[QUICK_START.md](QUICK_START.md)** - Quick start guide with examples
- **[FIXES_SUMMARY.md](FIXES_SUMMARY.md)** - Detailed improvements and fixes log
- **[CS_482_Final_Project.pdf](CS_482_Final_Project%20(1).pdf)** - Full research paper

## References

Please read our full paper on this project: [CS_482_Final_Project.pdf](CS_482_Final_Project%20(1).pdf)

## License

Academic project - see paper for citations and references.

## Contributors

See paper for full author list and acknowledgments.

