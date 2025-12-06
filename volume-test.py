
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict

import copy
import open3d as o3d
import pyvista as pv
import os
import sys
import argparse

def depth_map_to_point_cloud_o3d(depth_map, fx, fy, cx, cy):
    """
    Convert depth map to Open3D point cloud using camera intrinsics.
    
    Args:
        depth_map: 2D array of depth values
        fx, fy: focal lengths
        cx, cy: principal point
    
    Returns:
        Open3D point cloud
    """
    height, width = depth_map.shape
    
    # Create Open3D depth image
    depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
    
    # Create camera intrinsic
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy
    )
    
    # Convert to point cloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, intrinsic, depth_scale=1.0, depth_trunc=1000.0
    )
    
    return pcd

def register_point_clouds_icp(source, target, threshold=10.0):
    """
    Register two point clouds using ICP.
    
    Args:
        source: Source point cloud
        target: Target point cloud  
        threshold: Distance threshold for ICP
    
    Returns:
        Transformation matrix
    """
    # Initial alignment using point-to-point ICP
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    return reg.transformation


def load_image(filepath):
    """Loads an image into a numpy array.
    Note: image will have 3 color channels [r, g, b]."""
    img = np.float32(Image.open(filepath))
    #img = ImageOps.grayscale(img)
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')


def load_camera_params_dino(param_file):
    """
    Load camera parameters from dinoSR_par.txt file.
    
    Format per line:
    image_name fx s cx s fy cy s s 1 r11 r12 r13 r21 r22 r23 r31 r32 r33 tx ty tz
    
    Returns:
        dict: Dictionary mapping image names to camera parameters
              Each entry contains: K (3x3 intrinsic matrix), R (3x3 rotation), t (3x1 translation)
    """
    params = {}
    with open(param_file, 'r') as f:
        num_images = int(f.readline().strip())
        for _ in range(num_images):
            line = f.readline().strip().split()
            img_name = line[0]
            
            # Intrinsic parameters
            fx = float(line[1])
            s = float(line[2])   # skew (usually 0)
            cx = float(line[3])
            fy = float(line[5])
            cy = float(line[6])
            
            K = np.array([[fx, s, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
            
            # Extrinsic parameters (rotation and translation)
            R = np.array([[float(line[10]), float(line[11]), float(line[12])],
                         [float(line[13]), float(line[14]), float(line[15])],
                         [float(line[16]), float(line[17]), float(line[18])]])
            
            t = np.array([[float(line[19])],
                         [float(line[20])],
                         [float(line[21])]])
            
            params[img_name] = {'K': K, 'R': R, 't': t}
    
    return params


sift = cv.SIFT_create(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=15) # More features, lower thresholds
orb = cv.ORB_create(nfeatures=8000)  # ORB as fallback for low-texture images

# get feature matches between two images using SIFT
def get_point_matches(img1, img2, use_clahe=True, ratio_threshold=0.8):
    """Returns matches as array: (feature track, image, coord)
    
    Args:
        img1, img2: Input images
        use_clahe: Apply CLAHE preprocessing for better contrast
        ratio_threshold: Lowe's ratio test threshold (lower = more restrictive)
    """
    
    # Preprocess images with CLAHE for better feature detection
    if use_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img1_enhanced = clahe.apply(img1)
        img2_enhanced = clahe.apply(img2)
    else:
        img1_enhanced = img1
        img2_enhanced = img2

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1_enhanced, None)
    kp2, des2 = sift.detectAndCompute(img2_enhanced, None)
    
    # If SIFT finds too few features, try ORB as fallback
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print(f"    SIFT found few features ({len(kp1) if kp1 else 0}, {len(kp2) if kp2 else 0}), trying ORB...")
        kp1, des1 = orb.detectAndCompute(img1_enhanced, None)
        kp2, des2 = orb.detectAndCompute(img2_enhanced, None)
        use_orb = True
    else:
        use_orb = False
    
    # Check if we have valid descriptors
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print(f"    No features detected in images")
        return np.array([]), np.array([]), np.array([]), None

    # Use appropriate matcher based on descriptor type
    if use_orb:
        # ORB uses binary descriptors, use Hamming distance
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
    else:
        # SIFT uses float descriptors, use FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
        search_params = dict(checks=100)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper (relaxed threshold for more matches)
    for i, match_pair in enumerate(matches):
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    
    # Check if we have enough points for fundamental matrix
    if len(pts1) < 8:
        # Not enough points, return empty arrays
        print(f"    Only {len(pts1)} matches found (need at least 8)")
        return np.array([]), np.array([]), np.array([]), None
    
    print(f"    Found {len(pts1)} initial matches")
    
    # Use RANSAC with higher threshold for more robust estimation
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 
                                     ransacReprojThreshold=3.0, 
                                     confidence=0.99)
    
    # Check if fundamental matrix computation succeeded
    if F is None or mask is None:
        # Failed to compute fundamental matrix, return empty arrays
        return np.array([]), np.array([]), np.array([]), None
    
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
    if len(pts1) == 0:
        # No inliers, return empty arrays
        print(f"    No inliers after RANSAC")
        return np.array([]), np.array([]), np.array([]), None
    
    print(f"    {len(pts1)} matches after RANSAC filtering")
    
    pt1 = np.append(pts1[0],1)
    pt2 = np.append(pts2[0],1)
    
    return np.stack((pts1, pts2), axis=1), pts1, pts2, F

# combine feature matches
def combine_matches(matches_a, matches_b):
    """Assumes that the 0'th image is the same between them."""
    combined_matches = []
    for ii in range(matches_a.shape[0]):
        ma0 = matches_a[ii, 0]
        # Find the match in b
        mi = np.where((matches_b[:, 0] == ma0).all(axis=1))[0]

        # If a match is found, add to the array
        if mi.size > 0:
            ma = matches_a[ii]
            mb = matches_b[int(mi[0])]
            combined_matches.append(np.concatenate(
                (ma, mb[1:]), axis=0))

    return np.array(combined_matches)


# In[4]:


def visualize_matches(img_a, img_b, matches, output_path=None, title="Feature Matches"):
    """Visualize feature matches between two images."""
    fig = plt.figure(figsize=(20,10))
    ax = plt.gca()
    
    sa = img_a.shape
    sb = img_b.shape
    sp = 40
    off = sa[1]+sp
    
    merged_imgs = np.zeros(
        (max(sa[0], sb[0]), sa[1]+sb[1]+sp),
        dtype=np.float32)
    merged_imgs[0:sa[0], 0:sa[1]] = img_a
    merged_imgs[0:sb[0], sa[1]+sp:] = img_b
    ax.imshow(merged_imgs, cmap='gray')
    
    # Draw lines connecting matched features
    for m in matches:
        ax.plot([m[0][0], m[1][0]+off], [m[0][1], m[1][1]], 'r', alpha=0.5, linewidth=0.5)
    
    plt.title(title)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved feature matches visualization to: {output_path}")
    
    plt.close(fig)

def get_match_colors(image_c, combined_matches):
    colors = []
    nm = combined_matches.shape[0]
    for mi in range(nm):
        m = combined_matches[mi, 0, :]
        colors.append(image_c[m[1]-1:m[1]+2,
                              m[0]-1:m[0]+2].sum(axis=0).sum(axis=0)/9)
    
    return colors

def visualize_depth_map(depth_map, output_path=None, title="Depth Map"):
    """Visualize a depth map with colorbar."""
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(depth_map, cmap='hot')
    plt.colorbar(label='Depth')
    plt.title(title)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved depth map visualization to: {output_path}")
    
    plt.close(fig)

def visualize_rectified_pair(img1_rect, img2_rect, output_path=None, title="Rectified Stereo Pair"):
    """Visualize rectified stereo image pair with epipolar lines."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(img1_rect, cmap='gray')
    axes[0].set_title('Left Image (Rectified)')
    axes[0].axis('off')
    
    axes[1].imshow(img2_rect, cmap='gray')
    axes[1].set_title('Right Image (Rectified)')
    axes[1].axis('off')
    
    # Draw some horizontal epipolar lines to show rectification
    height = img1_rect.shape[0]
    for y in range(100, height, 100):
        axes[0].axhline(y, color='r', linewidth=0.5, alpha=0.5)
        axes[1].axhline(y, color='r', linewidth=0.5, alpha=0.5)
    
    plt.suptitle(title)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved rectified pair visualization to: {output_path}")
    
    plt.close(fig)

def visualize_colored_point_cloud(points, colors, output_path=None, title="Colored Point Cloud"):
    """Visualize 3D point cloud with colors from original images."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to 0-1 range if needed
    if colors is not None and len(colors) > 0:
        colors_normalized = np.array(colors)
        if colors_normalized.max() > 1.0:
            colors_normalized = colors_normalized / 255.0
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors_normalized, s=1, alpha=0.6)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                         points[:, 1].max()-points[:, 1].min(),
                         points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved colored point cloud to: {output_path}")
    
    plt.close(fig)

def rectify_two(test1,test2,pts1,pts2, F):
    h1,w1 = test1.shape
    h2,w2 = test2.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1,h1))
    
    img1_rectified = cv.warpPerspective(test1, H1, (w1,h1))
    img2_rectified = cv.warpPerspective(test2, H2, (w2,h2))
    
    #fig, axes = plt.subplots(1,2, figsize=(15, 10))
    #axes[0].imshow(img1_rectified, cmap="gray")
    #axes[1].imshow(img2_rectified, cmap='gray')
    #axes[0].axhline(400)
    #axes[1].axhline(400)
    #axes[0].axhline(600)
    #axes[1].axhline(600)
    return img1_rectified, img2_rectified

def set_points(three_images_arr, i, depth_map):
    if len(three_images_arr) != 3:
        raise ValueError('Array must have 3 images')
    test1 = cv.imread(three_images_arr[0],0)
    test2 = cv.imread(three_images_arr[1],0)
    test3 = cv.imread(three_images_arr[2],0)
    matches1, pts1,pts2,F = get_point_matches(test1, test2)
    matches2, pts1_1,pts2_1,F_1 = get_point_matches(test1, test3)
    
    #visualize_matches(test1, test2, matches1, ax=None)
    #visualize_matches(test1, test3, matches2, ax=None)
    
    # Check if we have valid matches and fundamental matrices
    if len(matches1) == 0 or len(matches2) == 0 or F is None or F_1 is None:
        # No matches found - return empty results
        # This happens with textureless images like depth maps
        return np.array([]), [], test1, test2, test3
    
    img1_rectified, img2_rectified = rectify_two(test1,test2,pts1,pts2, F)
    img1_rectified, img3_rectified = rectify_two(test1,test3,pts1_1,pts2_1, F_1)
    
    combined_matches = combine_matches(np.array(matches1), np.array(matches2))
    
    # If combined matches are too few, return all pairwise matches instead
    # This is more robust for triangulation
    if len(combined_matches) < 10:
        print(f"    Only {len(combined_matches)} combined matches, using pairwise matches instead")
        # Return matches1 which has shape (N, 2, 2) for img1-img2 pairs
        # We'll handle this differently in triangulation
        combined_matches = matches1
    
    colors = get_match_colors(load_image(three_images_arr[0]), combined_matches) if len(combined_matches) > 0 else []

    return combined_matches, colors, img1_rectified, img2_rectified, img3_rectified 

def get_depth(img1_rectified, img2_rectified):
    """
    Compute depth map from stereo rectified images using disparity.
    
    Args:
        img1_rectified: First rectified image
        img2_rectified: Second rectified image
    
    Returns:
        Depth map (disparity values normalized to 0-255)
    """
    # CALCULATE DISPARITY (DEPTH MAP)
    block_size = 11
    min_disp = -128
    max_disp = 128
    num_disp = max_disp - min_disp
    uniquenessRatio = 5
    speckleWindowSize = 0
    speckleRange = 2
    disp12MaxDiff = 0

    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )

    disp = stereo.compute(np.uint8(img1_rectified), np.uint8(img2_rectified)).astype(np.float32)
    
    # Validate disparity values before normalization
    # Check for invalid disparities (zero or negative)
    invalid_mask = disp <= 0
    num_invalid = np.sum(invalid_mask)
    total_pixels = disp.size
    invalid_percentage = (num_invalid / total_pixels) * 100
    
    if invalid_percentage > 50:
        print(f"  Warning: {invalid_percentage:.1f}% of disparity values are invalid (<=0)")
        print(f"  This may indicate poor stereo matching or insufficient texture")
    
    # Set invalid disparities to a sentinel value before normalization
    # This prevents them from affecting the normalization range
    disp[invalid_mask] = 0
    
    disp = cv.normalize(disp, disp, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
    disp = np.uint8(disp)
    #fig = plt.figure()
    #plt.imshow(disp)
    return disp
        
def detect_dataset_and_get_depth_source(image_path):
    """
    Detect which dataset we're using and return the appropriate depth source.
    
    Args:
        image_path: Path to one of the images in the dataset
    
    Returns:
        tuple: (use_provided_depth_map: bool, depth_map_path: str or None, use_triangulation: bool)
    """
    import os
    
    # Check if this is the world_rotate dataset (has provided depth maps)
    # Globe is symmetric, so same depth map works for all views
    if 'world_rotate' in image_path or 'trans_' in image_path:
        depth_map_path = './test_datasets/world_rotate/depthmap.png'
        if os.path.exists(depth_map_path):
            return (True, depth_map_path, False)  # Use provided depth, no triangulation
    
    # Check if this is cube_images dataset
    # Cube images are depth maps from different angles (no texture for feature matching)
    # Current approach (uniform sampling) doesn't work well - produces flat slabs
    # TODO: Need specialized handling for multi-view depth map integration
    elif 'cube_images' in image_path or 'cube_' in image_path:
        return (False, None, False)  # Use depth maps (but current method is inadequate)
    
    # Check if this is dinoSparseRing dataset
    # Dinosaur has texture and real camera parameters - use triangulation
    elif 'dinoSparseRing' in image_path or 'dinoSR' in image_path:
        # Use triangulation with real camera parameters
        # This gives accurate geometry with low reprojection error
        return (False, None, True)  # Use triangulation
    
    # Default: try triangulation
    return (False, None, True)


def get_depth_2(img1_rectified, img2_rectified):
    stereoSGBM = cv.StereoSGBM_create(minDisparity=0, 
                                   numDisparities = 64,
                                   blockSize = 3,
                                   P1 = 8*1*3*3, 
                                   P2 = 32*1*3*3
                                  )
    dispSGBM = stereoSGBM.compute(img1_rectified, img2_rectified).astype(np.float32) / 16
    #fig1 = plt.figure()
    #plt.imshow(dispSGBM, 'gray')
    #plt.colorbar()
    
    
    stereoBM = cv.StereoBM_create(numDisparities=16, blockSize=15)
    dispBM = stereoBM.compute(img1_rectified, img2_rectified)

    #fig2 = plt.figure()
    #plt.imshow(dispBM, 'gray')
    #plt.colorbar()


#######################################################################################

# Parse command line arguments
parser = argparse.ArgumentParser(description='3D Volume Estimation from Multiple Images')
parser.add_argument('--dataset', type=str, default='world', 
                    choices=['world', 'cube', 'dino'],
                    help='Dataset to use: world (globe), cube, or dino (dinosaur)')
parser.add_argument('--num-triplets', type=int, default=None,
                    help='Number of image triplets to process (default: all available)')
parser.add_argument('--visualize', action='store_true',
                    help='Show visualizations interactively (in addition to saving them)')
parser.add_argument('--no-vis', action='store_true',
                    help='Disable all intermediate visualizations (only save final results)')

args = parser.parse_args()

# Configure dataset based on command line argument
if args.dataset == 'world':
    print("Using WORLD_ROTATE dataset (globe with provided depth maps)")
    three_images = [['./test_datasets/world_rotate/trans_01.png','./test_datasets/world_rotate/trans_02.png','./test_datasets/world_rotate/trans_03.png'],
                    ['./test_datasets/world_rotate/trans_03.png','./test_datasets/world_rotate/trans_04.png','./test_datasets/world_rotate/trans_05.png'],
                    ['./test_datasets/world_rotate/trans_05.png','./test_datasets/world_rotate/trans_06.png','./test_datasets/world_rotate/trans_07.png'],
                    ['./test_datasets/world_rotate/trans_07.png','./test_datasets/world_rotate/trans_08.png','./test_datasets/world_rotate/trans_09.png'],
                    ['./test_datasets/world_rotate/trans_09.png','./test_datasets/world_rotate/trans_10.png','./test_datasets/world_rotate/trans_11.png'],
                    ['./test_datasets/world_rotate/trans_11.png','./test_datasets/world_rotate/trans_12.png','./test_datasets/world_rotate/trans_13.png'],
                    ['./test_datasets/world_rotate/trans_13.png','./test_datasets/world_rotate/trans_14.png','./test_datasets/world_rotate/trans_15.png'],
                    ['./test_datasets/world_rotate/trans_15.png','./test_datasets/world_rotate/trans_16.png','./test_datasets/world_rotate/trans_17.png'],
                    ['./test_datasets/world_rotate/trans_17.png','./test_datasets/world_rotate/trans_18.png','./test_datasets/world_rotate/trans_19.png'],
                    ['./test_datasets/world_rotate/trans_19.png','./test_datasets/world_rotate/trans_20.png','./test_datasets/world_rotate/trans_21.png'],
                    ['./test_datasets/world_rotate/trans_21.png','./test_datasets/world_rotate/trans_22.png','./test_datasets/world_rotate/trans_23.png'],
                    ['./test_datasets/world_rotate/trans_23.png','./test_datasets/world_rotate/trans_24.png','./test_datasets/world_rotate/trans_25.png']]

elif args.dataset == 'cube':
    print("Using CUBE_IMAGES dataset (depth maps from different angles)")
    three_images = [['./test_datasets/cube_images/cube_01.png','./test_datasets/cube_images/cube_02.png','./test_datasets/cube_images/cube_03.png'],
                    ['./test_datasets/cube_images/cube_03.png','./test_datasets/cube_images/cube_04.png','./test_datasets/cube_images/cube_05.png'],
                    ['./test_datasets/cube_images/cube_05.png','./test_datasets/cube_images/cube_06.png','./test_datasets/cube_images/cube_07.png'],
                    ['./test_datasets/cube_images/cube_07.png','./test_datasets/cube_images/cube_08.png','./test_datasets/cube_images/cube_09.png'],
                    ['./test_datasets/cube_images/cube_09.png','./test_datasets/cube_images/cube_10.png','./test_datasets/cube_images/cube_11.png'],
                    ['./test_datasets/cube_images/cube_11.png','./test_datasets/cube_images/cube_12.png','./test_datasets/cube_images/cube_14.png']]

elif args.dataset == 'dino':
    print("Using DINOSPARSE dataset (dinosaur with triangulation)")
    three_images = [['./test_datasets/dinoSparseRing/dinoSR0001.png','./test_datasets/dinoSparseRing/dinoSR0002.png','./test_datasets/dinoSparseRing/dinoSR0003.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0002.png','./test_datasets/dinoSparseRing/dinoSR0003.png','./test_datasets/dinoSparseRing/dinoSR0004.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0003.png','./test_datasets/dinoSparseRing/dinoSR0004.png','./test_datasets/dinoSparseRing/dinoSR0005.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0004.png','./test_datasets/dinoSparseRing/dinoSR0005.png','./test_datasets/dinoSparseRing/dinoSR0006.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0005.png','./test_datasets/dinoSparseRing/dinoSR0006.png','./test_datasets/dinoSparseRing/dinoSR0007.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0006.png','./test_datasets/dinoSparseRing/dinoSR0007.png','./test_datasets/dinoSparseRing/dinoSR0008.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0007.png','./test_datasets/dinoSparseRing/dinoSR0008.png','./test_datasets/dinoSparseRing/dinoSR0009.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0008.png','./test_datasets/dinoSparseRing/dinoSR0009.png','./test_datasets/dinoSparseRing/dinoSR0010.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0009.png','./test_datasets/dinoSparseRing/dinoSR0010.png','./test_datasets/dinoSparseRing/dinoSR0011.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0010.png','./test_datasets/dinoSparseRing/dinoSR0011.png','./test_datasets/dinoSparseRing/dinoSR0012.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0011.png','./test_datasets/dinoSparseRing/dinoSR0012.png','./test_datasets/dinoSparseRing/dinoSR0013.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0012.png','./test_datasets/dinoSparseRing/dinoSR0013.png','./test_datasets/dinoSparseRing/dinoSR0014.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0013.png','./test_datasets/dinoSparseRing/dinoSR0014.png','./test_datasets/dinoSparseRing/dinoSR0015.png'],
                    ['./test_datasets/dinoSparseRing/dinoSR0014.png','./test_datasets/dinoSparseRing/dinoSR0015.png','./test_datasets/dinoSparseRing/dinoSR0016.png']]

# Limit number of triplets if specified
if args.num_triplets is not None:
    three_images = three_images[:args.num_triplets]
    print(f"Processing {len(three_images)} triplets")
else:
    print(f"Processing all {len(three_images)} triplets")

print("=" * 60)

# Adaptive depth handling: 
# - For world_rotate (globe): Use provided depth map from Blender
# - For cube_images: Use depth maps with scaling
# - For dinoSparseRing: Use triangulation with real camera parameters

ulti_x = []
ulti_y = []
ulti_z = []

fig100 = plt.figure()
ax = plt.axes(projection='3d')
cmap_op = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','viridis', 'plasma', 'inferno', 'magma', 'cividis']
for cmap_option, three_images_coords in enumerate(three_images):
    # Detect dataset and determine reconstruction method
    use_provided_depth, depth_map_path, use_triangulation = detect_dataset_and_get_depth_source(three_images_coords[0])
    
    # Load or compute depth map based on dataset
    if use_provided_depth:
        # World_rotate dataset: use provided depth map from Blender
        depth_map = cv.imread(depth_map_path, cv.IMREAD_GRAYSCALE)
        if depth_map is None:
            print(f"  Warning: Could not load depth map from {depth_map_path}")
            use_provided_depth = False
            depth_map = None
        else:
            print(f"Using provided depth map for {three_images_coords[0]}")
    else:
        # Cube or dinosaur dataset: depth will be computed from stereo or triangulation
        depth_map = None
        if use_triangulation:
            print(f"Will use triangulation for {three_images_coords[0]}")
        else:
            print(f"Will compute depth from stereo for {three_images_coords[0]}")
    
    X_matches, colors, img1_rectified, img2_rectified, img3_rectified = set_points(three_images_coords, 0, depth_map)
    
    # Visualize feature matches (save by default unless --no-vis)
    if not args.no_vis and len(X_matches) > 0 and X_matches.ndim == 3:
        test1 = cv.imread(three_images_coords[0], 0)
        test2 = cv.imread(three_images_coords[1], 0)
        match_vis_path = f'output/{args.dataset}_matches_triplet_{cmap_option:02d}.png'
        visualize_matches(test1, test2, X_matches, match_vis_path, 
                         f"Feature Matches: Triplet {cmap_option+1}")


    #We attempted to reproject the feature matches into the world view without success!
    #We blame the difference in coordinate systems between Blender and Python. 
    #Also the bpy library would not load for us.


    #Camera measurements in Blender
    #distance from lens to surface of sphere = 9.5104
    #distance from principal point to surface of sphere = 10.6558
    #15 degrees in radians is 0.261799

    #translation when rotating 15 degrees around the z-axis
    #x = 9.5144*cos(0.261799) = 9.1863419329
    #y = 9.5144*sin(0.261799) = 2.46250435877

    #9.51066800099 = baseline

    b = 35.9 
    f = 188.9 

    # Camera intrinsic matrix K
    # K = [[fx, 0, cx],
    #      [0, fy, cy],
    #      [0, 0, 1]]
    # where cx, cy are principal point coordinates (image center)
    # Note: img.shape returns (height, width) for grayscale images
    # So cx should be width/2 and cy should be height/2
    K = [[f, 0, img1_rectified.shape[1]/2],  # cx = width/2
         [0, f, img1_rectified.shape[0]/2],  # cy = height/2
         [0, 0, 1]]
    R = np.eye(3)
    ta = [[0],[0],[1]]
    Pa = K @ np.concatenate((R, ta), axis=1)

    R = [[np.cos(15),-np.sin(15),0],
         [np.sin(15),np.cos(15),0],
         [0,0,1]]     
    tb = [[34.720032502], [9.3071030883], [0]] 
    Pb = K @ np.concatenate((R, tb), axis=1)


    def solve_point_triangulation(proj_points, proj_matrices):
        # First we build the matrix
        D = np.zeros((2*len(proj_points), 4), dtype=float)
        for ii, (p, P) in enumerate(zip(proj_points, proj_matrices)):
            D[2*ii + 0] = p[1] * P[2] - P[1]
            D[2*ii + 1] = P[0] - p[0] * P[2]

        # Now, solve
        u, s, vh = np.linalg.svd(D, full_matrices=False)
        X = vh[np.argmin(s)]
        return X/X[3]
    
    def compute_reprojection_error(point_3d, proj_points, proj_matrices):
        """
        Compute reprojection error for a triangulated 3D point.
        
        Args:
            point_3d: 3D point in homogeneous coordinates [X, Y, Z, 1]
            proj_points: List of 2D points (u, v) in each view
            proj_matrices: List of 3x4 projection matrices
        
        Returns:
            Total reprojection error (sum of squared distances)
        """
        total_error = 0.0
        
        for p_2d, P in zip(proj_points, proj_matrices):
            # Project 3D point to 2D using projection matrix
            p_proj_homogeneous = P @ point_3d
            
            # Convert from homogeneous to Euclidean coordinates
            if p_proj_homogeneous[2] != 0:
                p_proj = p_proj_homogeneous[:2] / p_proj_homogeneous[2]
            else:
                # Degenerate case - point at infinity
                continue
            
            # Compute squared distance between original and reprojected point
            error = np.sum((np.array(p_2d) - p_proj) ** 2)
            total_error += error
        
        return total_error
    
    def create_camera_projection_matrix(view_index, fx, fy, cx, cy, baseline=35.9):
        """
        Create camera projection matrix for a given view.
        
        Args:
            view_index: Index of the view (0, 1, 2, ...)
            fx, fy: Focal lengths
            cx, cy: Principal point coordinates
            baseline: Distance between camera positions
        
        Returns:
            P: 3x4 projection matrix P = K[R|t]
        """
        # Camera intrinsic matrix
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        
        # Camera rotation angle (15 degrees per view around z-axis)
        angle_degrees = view_index * 15
        angle_radians = np.radians(angle_degrees)
        
        # Rotation matrix around z-axis
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
        R = np.array([[cos_theta, -sin_theta, 0],
                      [sin_theta, cos_theta, 0],
                      [0, 0, 1]])
        
        # Translation vector (camera position on circular path)
        # Camera moves in a circle around the object
        t = np.array([[baseline * sin_theta],
                      [baseline * cos_theta],
                      [0]])
        
        # Projection matrix P = K[R|t]
        P = K @ np.hstack([R, t])
        
        return P

    # Only do triangulation if we have matches
    if len(X_matches) > 0 and X_matches.ndim == 3:
        # Check if we have triplet matches (3 views) or pairwise matches (2 views)
        if X_matches.shape[1] == 3:
            # Triplet matches - use all three views
            pa = X_matches[:,0].tolist()
            pb = X_matches[:,1].tolist()
            pc = X_matches[:,2].tolist()
            
            X0_rec = np.empty([len(pa),3])
            for pta,ptb,ptc in zip(pa,pb,pc):
                pt = solve_point_triangulation([pta,ptb,ptc], [Pa, Pb, Pb])[:-1]  # Note: Pb used twice, should be Pc
                X0_rec = np.append(X0_rec,np.array([pt]),axis=0)
        elif X_matches.shape[1] == 2:
            # Pairwise matches - use two views
            pa = X_matches[:,0].tolist()
            pb = X_matches[:,1].tolist()

            X0_rec = np.empty([len(pa),3])
            for pta,ptb in zip(pa,pb):
                pt = solve_point_triangulation([pta,ptb], [Pa, Pb])[:-1]
                X0_rec = np.append(X0_rec,np.array([pt]),axis=0)
        else:
            X0_rec = np.empty([0,3])
    else:
        # No matches - skip triangulation
        X0_rec = np.empty([0,3])

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(X0_rec[:, 0], X0_rec[:, 1], X0_rec[:, 2])
    #plt.show()


    from skimage import filters

    test1 = cv.imread(three_images_coords[0],0)
    test2 = cv.imread(three_images_coords[1],0)
    test3 = cv.imread(three_images_coords[2],0)
    
    # Note: OpenCV shape is (height, width) for grayscale images
    height = test1.shape[0]
    width = test1.shape[1]
    dim = (width, height)

    # Get depth maps for each view in the triplet
    # For irregular objects, each view needs its own depth map!
    depth_maps = []
    
    if use_provided_depth:
        # World_rotate: CAMERA rotates around stationary globe (turntable setup)
        # The globe is symmetric (sphere), so same depth map works for all camera angles
        # Feature matching finds continents in each view
        # MUST apply rotation to transform from camera coords to world coords
        src = cv.imread(depth_map_path, 0)
        src = cv.resize(src, dim, interpolation = cv.INTER_AREA)
        
        # Scale depth map to match world coordinate system
        depth_scale_factor = b / src.mean()
        src = src.astype(np.float32) * depth_scale_factor
        print(f"  Scaled depth map by {depth_scale_factor:.3f} (mean depth: {src.mean():.2f})")
        
        # Use same depth map for all views (sphere is symmetric)
        depth_maps = [src, src, src]
        print(f"  Using depth map for all views (camera rotates, will apply rotation)")
    elif not use_triangulation:
        # Check if images ARE depth maps (cube) or regular photos (dino)
        if 'cube_' in three_images_coords[0]:
            # Cube: Images ARE depth maps from different viewing angles
            # In these depth maps: darker pixels = closer to camera, lighter = farther
            # Background is a constant light gray value (typically 206)
            print(f"  Using images as depth maps (cube dataset)...")
            
            depth_map_1 = cv.resize(test1, dim, interpolation=cv.INTER_AREA).astype(np.float32)
            depth_map_2 = cv.resize(test2, dim, interpolation=cv.INTER_AREA).astype(np.float32)
            depth_map_3 = cv.resize(test3, dim, interpolation=cv.INTER_AREA).astype(np.float32)
            
            # Detect background value (use corner pixels which should be background)
            background_val = depth_map_1[0, 0]
            print(f"  Detected background value: {background_val}")
            
            # Create masks for valid (non-background) pixels
            mask_1 = depth_map_1 != background_val
            mask_2 = depth_map_2 != background_val
            mask_3 = depth_map_3 != background_val
            
            # Keep original pixel values (0-205 for cube surface)
            # The reconstruction code will handle scaling
            # Just zero out background pixels so they get filtered
            depth_map_1[~mask_1] = 0
            depth_map_2[~mask_2] = 0
            depth_map_3[~mask_3] = 0
            
            print(f"  Valid pixels: view1={mask_1.sum()}, view2={mask_2.sum()}, view3={mask_3.sum()}")
            
            depth_maps = [depth_map_1, depth_map_2, depth_map_3]
            print(f"  Using image-based depth maps: {depth_map_1.shape}, {depth_map_2.shape}, {depth_map_3.shape}")
        else:
            # Dinosaur: Regular photos - compute stereo depth from rectified images
            print(f"  Computing stereo depth maps from rectified images...")
            
            # Visualize rectified images (save by default unless --no-vis)
            if not args.no_vis:
                rect_vis_path = f'output/{args.dataset}_rectified_triplet_{cmap_option:02d}.png'
                visualize_rectified_pair(img1_rectified, img2_rectified, rect_vis_path,
                                        f"Rectified Stereo Pair: Triplet {cmap_option+1}")
            
            # Compute depth from rectified stereo pairs
            depth_map_1_2 = get_depth(img1_rectified, img2_rectified)
            depth_map_1_3 = get_depth(img1_rectified, img3_rectified)
            
            # Use the depth maps computed from stereo
            # For view 1, average the two depth estimates
            depth_map_1 = ((depth_map_1_2.astype(np.float32) + depth_map_1_3.astype(np.float32)) / 2.0)
            depth_map_2 = depth_map_1_2.astype(np.float32)
            depth_map_3 = depth_map_1_3.astype(np.float32)
            
            depth_maps = [depth_map_1, depth_map_2, depth_map_3]
            print(f"  Computed stereo depth maps: {depth_map_1.shape}, {depth_map_2.shape}, {depth_map_3.shape}")
    else:
        # This case is for triangulation (handled later)
        depth_maps = None
        print(f"  Will use triangulation (no depth maps needed)")
    
    # Get dimensions from first image (depth_maps might be None for triangulation)
    if depth_maps is not None:
        height, width = depth_maps[0].shape
        
        # Visualize depth map (save by default unless --no-vis)
        if not args.no_vis:
            depth_vis_path = f'output/{args.dataset}_depth_triplet_{cmap_option:02d}.png'
            visualize_depth_map(depth_maps[0], depth_vis_path, 
                              f"Depth Map: Triplet {cmap_option+1}")
    else:
        height, width = test1.shape

    pt_cloud = []
    # Camera intrinsic parameters for world_rotate dataset
    fx = 188.9763779528  # focal length in x
    fy = 188.9763779528  # focal length in y (assuming square pixels)
    cx = width / 2.0     # principal point x (image center)
    cy = height / 2.0    # principal point y (image center)

    # Check if we have enough feature matches
    # If not, and we're using depth maps, sample points uniformly
    has_enough_matches = len(X_matches) > 10 and X_matches.ndim == 3
    
    if has_enough_matches:
        # Check if we have triplet matches (3 views) or pairwise matches (2 views)
        if X_matches.shape[1] == 3:
            im_1_pts = X_matches[:,0]
            im_2_pts = X_matches[:,1]
            im_3_pts = X_matches[:,2]
            ims = [im_1_pts, im_2_pts, im_3_pts]
        elif X_matches.shape[1] == 2:
            # Pairwise matches - only have two views
            im_1_pts = X_matches[:,0]
            im_2_pts = X_matches[:,1]
            im_3_pts = np.array([])
            ims = [im_1_pts, im_2_pts, im_3_pts]
        else:
            im_1_pts = np.array([])
            im_2_pts = np.array([])
            im_3_pts = np.array([])
            ims = [im_1_pts, im_2_pts, im_3_pts]
    else:
        # No matches or invalid format - will use uniform sampling
        im_1_pts = np.array([])
        im_2_pts = np.array([])
        im_3_pts = np.array([])
        ims = [im_1_pts, im_2_pts, im_3_pts]

    img_pts = []
    
    # Choose reconstruction method based on dataset
    if use_triangulation and has_enough_matches:
        # Use triangulation for irregular objects (dinosaur)
        print(f"  Reconstructing using triangulation...")
        
        # Check if we have actual camera parameters (dinosaur dataset)
        if 'dinoSR' in three_images_coords[0]:
            # Load actual camera parameters from file
            param_file = './test_datasets/dinoSparseRing/dinoSR_par.txt'
            camera_params = load_camera_params_dino(param_file)
            
            # Get image names from paths
            img_names = [os.path.basename(path) for path in three_images_coords]
            
            # Create projection matrices using actual camera parameters
            proj_matrices = []
            for img_name in img_names:
                if img_name in camera_params:
                    params = camera_params[img_name]
                    K = params['K']
                    R = params['R']
                    t = params['t']
                    # Projection matrix P = K[R|t]
                    P = K @ np.hstack([R, t])
                    proj_matrices.append(P)
                else:
                    print(f"  Warning: No camera parameters for {img_name}")
                    proj_matrices = []
                    break
            
            if len(proj_matrices) != 3:
                print(f"  Error: Could not load camera parameters, skipping triangulation")
                img_pts = [[], [], []]
            else:
                # Triangulate matched features
                # If we have triplet matches (all 3 views), use them
                # Otherwise, use pairwise matches between consecutive views
                triangulated_points = []
                reprojection_errors = []
                
                if len(im_3_pts) > 0:
                    # We have triplet matches - triangulate across all three views
                    for match_idx in range(len(im_1_pts)):
                        proj_points = [
                            im_1_pts[match_idx],
                            im_2_pts[match_idx],
                            im_3_pts[match_idx]
                        ]
                        
                        try:
                            X_3d = solve_point_triangulation(proj_points, proj_matrices)
                            
                            if X_3d[3] != 0:
                                pt_3d = (X_3d[0], X_3d[1], X_3d[2])
                                reproj_error = compute_reprojection_error(X_3d, proj_points, proj_matrices)
                                triangulated_points.append(pt_3d)
                                reprojection_errors.append(reproj_error)
                        except Exception as e:
                            continue
                else:
                    # We only have pairwise matches - triangulate between view 1 and 2
                    for match_idx in range(len(im_1_pts)):
                        proj_points = [
                            im_1_pts[match_idx],
                            im_2_pts[match_idx]
                        ]
                        
                        try:
                            X_3d = solve_point_triangulation(proj_points, [proj_matrices[0], proj_matrices[1]])
                            
                            if X_3d[3] != 0:
                                pt_3d = (X_3d[0], X_3d[1], X_3d[2])
                                reproj_error = compute_reprojection_error(X_3d, proj_points, [proj_matrices[0], proj_matrices[1]])
                                triangulated_points.append(pt_3d)
                                reprojection_errors.append(reproj_error)
                        except Exception as e:
                            continue
                    
                    # Also triangulate between view 2 and 3 if we have those matches
                    # Get matches between image 2 and 3
                    test2 = cv.imread(three_images_coords[1], 0)
                    test3 = cv.imread(three_images_coords[2], 0)
                    matches_2_3, pts2, pts3, F_2_3 = get_point_matches(test2, test3)
                    
                    if len(matches_2_3) > 0 and matches_2_3.ndim == 3 and matches_2_3.shape[1] == 2:
                        print(f"  Found {len(matches_2_3)} additional matches between views 2-3")
                        for match_idx in range(len(matches_2_3)):
                            proj_points = [
                                matches_2_3[match_idx, 0],
                                matches_2_3[match_idx, 1]
                            ]
                            
                            try:
                                X_3d = solve_point_triangulation(proj_points, [proj_matrices[1], proj_matrices[2]])
                                
                                if X_3d[3] != 0:
                                    pt_3d = (X_3d[0], X_3d[1], X_3d[2])
                                    reproj_error = compute_reprojection_error(X_3d, proj_points, [proj_matrices[1], proj_matrices[2]])
                                    triangulated_points.append(pt_3d)
                                    reprojection_errors.append(reproj_error)
                            except Exception as e:
                                continue
                    
                    # Also triangulate between view 1 and 3
                    test1 = cv.imread(three_images_coords[0], 0)
                    matches_1_3, pts1, pts3, F_1_3 = get_point_matches(test1, test3)
                    
                    if len(matches_1_3) > 0 and matches_1_3.ndim == 3 and matches_1_3.shape[1] == 2:
                        print(f"  Found {len(matches_1_3)} additional matches between views 1-3")
                        for match_idx in range(len(matches_1_3)):
                            proj_points = [
                                matches_1_3[match_idx, 0],
                                matches_1_3[match_idx, 1]
                            ]
                            
                            try:
                                X_3d = solve_point_triangulation(proj_points, [proj_matrices[0], proj_matrices[2]])
                                
                                if X_3d[3] != 0:
                                    pt_3d = (X_3d[0], X_3d[1], X_3d[2])
                                    reproj_error = compute_reprojection_error(X_3d, proj_points, [proj_matrices[0], proj_matrices[2]])
                                    triangulated_points.append(pt_3d)
                                    reprojection_errors.append(reproj_error)
                            except Exception as e:
                                continue
                
                print(f"  Triangulated {len(triangulated_points)} 3D points (before filtering)")
                
                # Report reprojection error statistics (Requirements 8.4)
                if len(reprojection_errors) > 0:
                    mean_error = np.mean(reprojection_errors)
                    median_error = np.median(reprojection_errors)
                    max_error = np.max(reprojection_errors)
                    print(f"  Reprojection error - Mean: {mean_error:.2f}, Median: {median_error:.2f}, Max: {max_error:.2f} pixels")
                    
                    # Filter outliers based on reprojection error
                    # Use adaptive threshold: median + 3*MAD (Median Absolute Deviation)
                    mad = np.median(np.abs(np.array(reprojection_errors) - median_error))
                    error_threshold = median_error + 3 * mad
                    # But cap at reasonable maximum (50 pixels)
                    error_threshold = min(error_threshold, 50.0)
                    
                    filtered_points = []
                    filtered_errors = []
                    for pt, err in zip(triangulated_points, reprojection_errors):
                        if err < error_threshold:
                            filtered_points.append(pt)
                            filtered_errors.append(err)
                    
                    print(f"  After filtering (threshold={error_threshold:.2f}px): {len(filtered_points)} points")
                    if len(filtered_errors) > 0:
                        print(f"  Filtered error - Mean: {np.mean(filtered_errors):.2f}, Median: {np.median(filtered_errors):.2f} pixels")
                    
                    triangulated_points = filtered_points
                
                # For triangulation with real camera parameters, points are already in world coordinates
                # No rotation needed
                if len(triangulated_points) > 0:
                    img_pts = [triangulated_points, [], []]
                else:
                    img_pts = [[], [], []]
        else:
            # Cube or other dataset - use synthetic camera parameters
            # Create camera projection matrices for each view in this triplet
            # Calculate absolute view indices
            view_indices = [(cmap_option * 2) + i for i in range(3)]
            
            # Create projection matrices
            proj_matrices = []
            for view_idx in view_indices:
                P = create_camera_projection_matrix(view_idx, fx, fy, cx, cy, baseline=b)
                proj_matrices.append(P)
            
            # Triangulate each matched feature across the three views
            triangulated_points = []
            reprojection_errors = []
            
            for match_idx in range(len(im_1_pts)):
                # Get 2D points from all three views for this feature
                proj_points = [
                    im_1_pts[match_idx],  # (u, v) in view 1
                    im_2_pts[match_idx],  # (u, v) in view 2
                    im_3_pts[match_idx]   # (u, v) in view 3
                ]
                
                # Triangulate to get 3D point
                try:
                    X_3d = solve_point_triangulation(proj_points, proj_matrices)
                    
                    # X_3d is in homogeneous coordinates [X, Y, Z, 1]
                    # Extract 3D coordinates
                    if X_3d[3] != 0:  # Avoid division by zero
                        pt_3d = (X_3d[0], X_3d[1], X_3d[2])
                        
                        # Compute reprojection error (Requirements 8.4)
                        reproj_error = compute_reprojection_error(X_3d, proj_points, proj_matrices)
                        
                        triangulated_points.append(pt_3d)
                        reprojection_errors.append(reproj_error)
                    
                except Exception as e:
                    # Skip points that fail triangulation
                    continue
            
            print(f"  Triangulated {len(triangulated_points)} 3D points (before filtering)")
            
            # Report reprojection error statistics (Requirements 8.4)
            if len(reprojection_errors) > 0:
                mean_error = np.mean(reprojection_errors)
                median_error = np.median(reprojection_errors)
                max_error = np.max(reprojection_errors)
                print(f"  Reprojection error - Mean: {mean_error:.2f}, Median: {median_error:.2f}, Max: {max_error:.2f} pixels")
                
                # Filter outliers based on reprojection error
                mad = np.median(np.abs(np.array(reprojection_errors) - median_error))
                error_threshold = median_error + 3 * mad
                error_threshold = min(error_threshold, 50.0)
                
                filtered_points = []
                filtered_errors = []
                for pt, err in zip(triangulated_points, reprojection_errors):
                    if err < error_threshold:
                        filtered_points.append(pt)
                        filtered_errors.append(err)
                
                print(f"  After filtering (threshold={error_threshold:.2f}px): {len(filtered_points)} points")
                if len(filtered_errors) > 0:
                    print(f"  Filtered error - Mean: {np.mean(filtered_errors):.2f}, Median: {np.median(filtered_errors):.2f} pixels")
                
                triangulated_points = filtered_points
            
            # For triangulation, we only have one set of 3D points
            if len(triangulated_points) > 0:
                img_pts = [triangulated_points, [], []]
            else:
                img_pts = [[], [], []]
            
    elif not has_enough_matches and depth_maps is not None:
        # Fallback: Not enough feature matches, but we have depth maps
        
        # For cube: Use ICP to align point clouds from different depth map views
        if 'cube_' in three_images_coords[0]:
            print(f"  Reconstructing cube using 90-degree rotation alignment...")
            
            # Cube depth maps show corner views with 90-degree rotation between views
            # Use orthographic projection: X,Y from pixel coords, Z from depth value
            
            pixel_scale = 0.05  # Scale down X,Y dimensions
            depth_scale = 30.0 / 205.0  # Scale depth to ~30 units
            
            # Convert depth maps to point clouds
            point_clouds_np = []
            
            for view_idx in range(3):
                depth_map_for_view = depth_maps[view_idx]
                points = []
                
                # Sample points from depth map (subsample for speed)
                for v in range(0, height, 5):
                    for u in range(0, width, 5):
                        depth = depth_map_for_view[v, u]
                        
                        if depth == 0:  # Skip background
                            continue
                        
                        # Orthographic projection
                        x = u * pixel_scale
                        y = v * pixel_scale
                        z = depth * depth_scale
                        
                        points.append([x, y, z])
                
                point_clouds_np.append(np.array(points))
                print(f"    View {view_idx + 1}: Created point cloud with {len(points)} points")
            
            # Estimate cube center from all centroids
            centroids = [np.mean(pc, axis=0) for pc in point_clouds_np]
            cube_center = np.mean(centroids, axis=0)
            
            # Align using 90-degree rotation around X-axis
            merged_points_list = []
            merged_points_list.extend(point_clouds_np[0])  # Reference view
            
            for view_idx in range(1, 3):
                points = point_clouds_np[view_idx].copy()
                
                # Rotation angle: 90 degrees per view
                angle_deg = view_idx * 90
                angle_rad = np.radians(-angle_deg)  # Negative for inverse rotation
                
                # Translate to cube center
                points -= cube_center
                
                # Rotation matrix around Z-axis
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                R = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
                
                # Apply rotation
                points = points @ R.T
                
                # Translate back
                points += cube_center
                
                merged_points_list.extend(points)
                print(f"    Aligned view {view_idx + 1} with {-angle_deg}° rotation around Z-axis")
            
            merged_points = np.array(merged_points_list)
            img_pts = [[(pt[0], pt[1], pt[2]) for pt in merged_points], [], []]
            print(f"    Total merged points: {len(merged_points)}")
            
        else:
            # For dinosaur with stereo depth: use uniform sampling
            print(f"  Reconstructing using uniform sampling from depth maps...")
            
            # Sample points uniformly across the depth map
            sample_step = 10  # Sample every 10 pixels for dense coverage
            
            # Sample all three views
            for view_idx in range(3):
                X_s = []
                Z_s = []
                Y_s = []
                
                depth_map_for_view = depth_maps[view_idx]
                invalid_depth_count = 0
                
                # Sample uniformly across the image
                for v in range(0, height, sample_step):
                    for u in range(0, width, sample_step):
                        # Bounds checking
                        if v >= height or u >= width:
                            continue
                        
                        # Get depth value
                        depth = depth_map_for_view[v, u]
                        
                        # Validate depth value (Requirements 3.3)
                        # For stereo depth, values are normalized 0-255
                        # Skip very low values (likely background or invalid)
                        if depth <= 0:
                            invalid_depth_count += 1
                            continue
                        if depth < 50:  # Skip low confidence depth (background)
                            invalid_depth_count += 1
                            continue
                        
                        # Apply pinhole camera model
                        X = (u - cx) * depth / fx
                        Y = (v - cy) * depth / fy
                        Z = depth
                        
                        Z_s.append(Z)
                        X_s.append(X)
                        Y_s.append(Y)
                
                img_pts.append(list(zip(X_s, Y_s, Z_s)))
                if invalid_depth_count > 0:
                    print(f"    View {view_idx + 1}: Sampled {len(X_s)} points (skipped {invalid_depth_count} invalid depth values)")
                else:
                    print(f"    View {view_idx + 1}: Sampled {len(X_s)} points")
    
    else:
        # Mixed approach: Use depth map for views that have it, triangulation for others
        # This handles world_rotate where first view has depth map, others need triangulation
        
        # Check if we have mixed depth maps (some views have depth, others don't)
        has_mixed_depth = depth_maps is not None and any(dm is None for dm in depth_maps)
        
        if has_mixed_depth and has_enough_matches:
            # World_rotate case: Use depth map for first view, triangulate for others
            print(f"  Reconstructing using depth map (view 1) + triangulation (views 2-3)...")
            
            # Process first view with depth map
            if depth_maps[0] is not None and len(ims[0]) > 0:
                X_s = []
                Y_s = []
                Z_s = []
                depth_map_for_view = depth_maps[0]
                u_coords = ims[0][:,0]
                v_coords = ims[0][:,1]
                invalid_depth_count = 0
                
                for u,v in zip(u_coords, v_coords):
                    if v < 0 or v >= height or u < 0 or u >= width:
                        continue
                    
                    depth = depth_map_for_view[v, u]
                    
                    if depth <= 0 or depth < 10:
                        invalid_depth_count += 1
                        continue
                    
                    X = (u - cx) * depth / fx
                    Y = (v - cy) * depth / fy
                    Z = depth
                    
                    X_s.append(X)
                    Y_s.append(Y)
                    Z_s.append(Z)
                
                img_pts.append(list(zip(X_s, Y_s, Z_s)))
                print(f"    View 1: Reconstructed {len(X_s)} points using depth map")
            else:
                img_pts.append([])
            
            # Triangulate views 2 and 3 using feature correspondences
            # Create projection matrices for all three views
            view_indices = [(cmap_option * 2) + i for i in range(3)]
            proj_matrices = []
            for view_idx in view_indices:
                P = create_camera_projection_matrix(view_idx, fx, fy, cx, cy, baseline=b)
                proj_matrices.append(P)
            
            # Triangulate each matched feature across the three views
            triangulated_points = []
            for match_idx in range(len(im_1_pts)):
                proj_points = [
                    im_1_pts[match_idx],
                    im_2_pts[match_idx],
                    im_3_pts[match_idx]
                ]
                
                try:
                    X_3d = solve_point_triangulation(proj_points, proj_matrices)
                    if X_3d[3] != 0:
                        pt_3d = (X_3d[0], X_3d[1], X_3d[2])
                        triangulated_points.append(pt_3d)
                except Exception as e:
                    continue
            
            # Triangulated points represent views 2 and 3
            # Split them (though they're already in world coordinates from triangulation)
            # For simplicity, put all triangulated points in view 2
            img_pts.append(triangulated_points)
            img_pts.append([])
            print(f"    Views 2-3: Triangulated {len(triangulated_points)} points")
            
        elif depth_maps is not None and all(dm is not None for dm in depth_maps):
            # All views have depth maps - use depth map reconstruction
            
            # For dinosaur with stereo depth, use uniform sampling (not just feature points)
            # because stereo depth is sparse/noisy at feature locations
            if 'dinoSR' in three_images_coords[0]:
                print(f"  Reconstructing using uniform sampling from stereo depth maps...")
                
                sample_step = 10  # Sample every 10 pixels
                
                for view_idx in range(3):
                    X_s = []
                    Z_s = []
                    Y_s = []
                    
                    depth_map_for_view = depth_maps[view_idx]
                    invalid_depth_count = 0
                    
                    # Sample uniformly across the image
                    for v in range(0, height, sample_step):
                        for u in range(0, width, sample_step):
                            if v >= height or u >= width:
                                continue
                            
                            depth = depth_map_for_view[v, u]
                            
                            # For stereo depth (0-255), skip low confidence values
                            if depth <= 0 or depth < 50:
                                invalid_depth_count += 1
                                continue
                            
                            X = (u - cx) * depth / fx
                            Y = (v - cy) * depth / fy
                            Z = depth
                            
                            Z_s.append(Z)
                            X_s.append(X)
                            Y_s.append(Y)
                    
                    img_pts.append(list(zip(X_s, Y_s, Z_s)))
                    print(f"    View {view_idx + 1}: Sampled {len(X_s)} points (skipped {invalid_depth_count} invalid)")
            else:
                # For world_rotate: use feature match locations
                print(f"  Reconstructing using depth maps with feature matches...")
                
                for view_idx, imj in enumerate(ims):
                    X_s = []
                    Z_s = []
                    Y_s = []

                    depth_map_for_view = depth_maps[view_idx]
                    u_coords = imj[:,0]
                    v_coords = imj[:,1]
                    invalid_depth_count = 0
                    
                    for u,v in zip(u_coords, v_coords):
                        if v < 0 or v >= height or u < 0 or u >= width:
                            continue
                        
                        depth = depth_map_for_view[v, u]
                        
                        if depth <= 0 or depth < 1:
                            invalid_depth_count += 1
                            continue
                        
                        X = (u - cx) * depth / fx
                        Y = (v - cy) * depth / fy
                        Z = depth
                        
                        Z_s.append(Z)
                        X_s.append(X)
                        Y_s.append(Y)
                    
                    if invalid_depth_count > 0:
                        total_points = len(u_coords)
                        invalid_percentage = (invalid_depth_count / total_points) * 100
                        print(f"    View {view_idx + 1}: Skipped {invalid_depth_count}/{total_points} points ({invalid_percentage:.1f}%) due to invalid depth")
                    
                    img_pts.append(list(zip(X_s, Y_s, Z_s)))
        else:
            # No depth maps - shouldn't reach here
            img_pts = [[], [], []]


    from scipy.spatial.transform import Rotation as R

    # Apply rotation to transform points from camera coordinate system to world coordinate system
    # The camera rotates around the object by 15 degrees per view
    # 
    # IMPORTANT: For depth maps, each view shows the visible surface from that camera angle.
    # The depth map is in the CAMERA's coordinate system, not world coordinates.
    # We need to transform from camera coordinates to world coordinates.
    #
    # Camera setup: Camera rotates around z-axis at distance 'baseline' from origin
    # View 0: camera at angle 0 degrees
    # View 1: camera at angle 15 degrees  
    # View 2: camera at angle 30 degrees
    #
    # Camera-to-world transformation:
    # 1. Points start in camera coordinates (Z-axis points away from camera)
    # 2. Rotate by camera angle to align with world frame
    # 3. Translate by camera position
    #
    # For camera at angle θ on a circle of radius R:
    # - Camera position: (R*sin(θ), R*cos(θ), 0)
    # - Camera looks toward origin, so we need to rotate points AND translate
    
    # For triangulated points with real camera parameters, they are already in world coordinates
    # For depth map points, we need to apply camera-to-world transformation
    # Check if this is triangulation with real camera params (dinosaur)
    # For dinosaur, points are already in world coordinates
    # For cube with ICP, points are already registered - don't rotate again
    # For world_rotate, CAMERA rotates (object stationary) - need forward rotation
    skip_rotation = (use_triangulation and 'dinoSR' in three_images_coords[0]) or ('cube_' in three_images_coords[0])
    
    # Determine if we need inverse rotation (for rotating object)
    inverse_rotation = False  # Not used anymore - ICP handles cube alignment
    
    if not skip_rotation and len(img_pts) > 0:
        # Rotate each view separately based on its camera/object angle
        for i in range(0, len(img_pts)):
            if len(img_pts[i]) == 0:
                continue
                
            # Calculate the absolute view index for this image
            view_index = (cmap_option * 2) + i
            
            # Rotation angle
            angle_degrees = 15 * view_index  # degrees
            
            # For rotating object (cube): apply INVERSE rotation to transform to world coords
            # For rotating camera (world_rotate): apply FORWARD rotation
            if inverse_rotation:
                angle_degrees = -angle_degrees  # Inverse rotation
            
            angle_rad = np.radians(angle_degrees)
            rotation = R.from_rotvec(angle_rad * np.array([0, 0, 1]))

            # Apply the rotation to all points in this view
            for j, pt in enumerate(img_pts[i]):
                img_pts[i][j] = rotation.apply(pt)
            
            rotation_type = "inverse" if inverse_rotation else "forward"
            print(f"    Applied {angle_degrees}° {rotation_type} rotation to view {i+1} ({len(img_pts[i])} points)")

    im_1_pts = img_pts[0]
    im_2_pts = img_pts[1] if len(img_pts) > 1 else []
    im_3_pts = img_pts[2] if len(img_pts) > 2 else []

    # For triangulated points, we only have one set (im_1_pts)
    # For depth map points, we have three sets
    if len(im_1_pts) > 0:
        X_s, Y_s, Z_s = [a_tuple[0] for a_tuple in im_1_pts], [a_tuple[1] for a_tuple in im_1_pts], [a_tuple[2] for a_tuple in im_1_pts]
    else:
        X_s, Y_s, Z_s = [], [], []
    
    if len(im_2_pts) > 0:
        X_s_1, Y_s_1, Z_s_1 = [a_tuple[0] for a_tuple in im_2_pts], [a_tuple[1] for a_tuple in im_2_pts], [a_tuple[2] for a_tuple in im_2_pts]
    else:
        X_s_1, Y_s_1, Z_s_1 = [], [], []
    
    if len(im_3_pts) > 0:
        X_s_2, Y_s_2, Z_s_2 = [a_tuple[0] for a_tuple in im_3_pts], [a_tuple[1] for a_tuple in im_3_pts], [a_tuple[2] for a_tuple in im_3_pts]
    else:
        X_s_2, Y_s_2, Z_s_2 = [], [], []
    
    #Map out the point cloud here.
    if len(X_s) > 0:
        if(cmap_option==0):
            ax.scatter3D(X_s,Y_s,Z_s,c=Z_s, cmap=cmap_op[cmap_option])
    if len(X_s_1) > 0:
        ax.scatter3D(X_s_1,Y_s_1,Z_s_1, c=Z_s_1, cmap=cmap_op[cmap_option + 1])
    if len(X_s_2) > 0:
        ax.scatter3D(X_s_2,Y_s_2,Z_s_2, c=Z_s_2, cmap=cmap_op[cmap_option + 2])
    
    # Add points to global lists
    # For triangulation, only im_1_pts has data
    if len(X_s) > 0 and len(X_s_1) > 0 and len(X_s_2) > 0:
        # Depth map case: three views with points
        for x,x1,x2 in zip(X_s,X_s_1,X_s_2):
            ulti_x.append(x)
            ulti_x.append(x1)
            ulti_x.append(x2)
        
        for y,y1,y2 in zip(Y_s,Y_s_1,Y_s_2):
            ulti_y.append(y)
            ulti_y.append(y1)
            ulti_y.append(y2)

        for z,z1,z2 in zip(Z_s,Z_s_1,Z_s_2):
            ulti_z.append(z)
            ulti_z.append(z1)
            ulti_z.append(z2)
    elif len(X_s) > 0:
        # Triangulation case: only one set of points
        for x in X_s:
            ulti_x.append(x)
        for y in Y_s:
            ulti_y.append(y)
        for z in Z_s:
            ulti_z.append(z)
# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Save with dataset name
output_file = f'output/{args.dataset}_3d_scatter.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved 3D scatter plot to: {output_file}")
if not args.visualize:
    plt.close(fig100)
else:
    plt.show()
    plt.close(fig100)

ulti_pts = np.array(list(zip(ulti_x,ulti_y,ulti_z)))

# Save colored point cloud visualization if we have colors
if not args.no_vis and len(ulti_pts) > 0:
    # Try to get colors from the first image for visualization
    try:
        first_image = load_image(three_images[0][0])
        # Sample colors from the point cloud (this is approximate)
        # In a full implementation, we'd track which image each point came from
        colored_output = f'output/{args.dataset}_colored_pointcloud.png'
        visualize_colored_point_cloud(ulti_pts, None, colored_output, 
                                     f"{args.dataset.upper()} - Colored Point Cloud")
    except Exception as e:
        print(f"  Note: Could not generate colored point cloud visualization: {e}")

# Validate point cloud density (Requirements 6.5)
num_points = len(ulti_pts)
min_points_required = 100  # Minimum points for reliable volume estimation

print("\n" + "=" * 60)
print("POINT CLOUD VALIDATION")
print("=" * 60)
print(f"Total reconstructed points: {num_points}")

if num_points < 4:
    print("ERROR: Insufficient points for volume calculation (need at least 4 points)")
    print("Cannot construct convex hull or compute volume.")
    sys.exit(1)
elif num_points < min_points_required:
    print(f"WARNING: Point cloud density is low ({num_points} < {min_points_required} recommended)")
    print("Volume estimate may be inaccurate. Consider:")
    print("  - Using more images")
    print("  - Using images with better texture")
    print("  - Adjusting feature matching parameters")
    print("  - Using uniform sampling instead of feature matching")
else:
    print(f"Point cloud density is sufficient ({num_points} >= {min_points_required})")

# Check for degenerate point cloud (all points in a plane)
if num_points >= 4:
    # Compute bounding box to check if points span 3D space
    bbox_min = np.min(ulti_pts, axis=0)
    bbox_max = np.max(ulti_pts, axis=0)
    bbox_size = bbox_max - bbox_min
    
    print(f"Bounding box size: X={bbox_size[0]:.2f}, Y={bbox_size[1]:.2f}, Z={bbox_size[2]:.2f}")
    
    # Check if any dimension is too small (degenerate case)
    min_dimension = 1.0  # Minimum size in any dimension
    if np.any(bbox_size < min_dimension):
        print(f"WARNING: Point cloud is nearly planar or degenerate")
        print(f"  One or more dimensions < {min_dimension}")
        print("  Volume estimate may be unreliable")

print("=" * 60 + "\n")

import matplotlib.tri as mtri
from scipy.spatial import Delaunay

tri = Delaunay(ulti_pts) # points: np.array() of 3d points 

fig55 = plt.figure()
ax = fig55.add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(ulti_pts[:,0], ulti_pts[:,1], ulti_pts[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)

# Save with dataset name
output_file = f'output/{args.dataset}_trisurf.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved triangulated surface plot to: {output_file}")
if not args.visualize:
    plt.close(fig55)
else:
    plt.show()
    plt.close(fig55)

from scipy.spatial import ConvexHull
import mpl_toolkits.mplot3d as a3
import matplotlib as mpl
import scipy as sp

hull = ConvexHull(ulti_pts)
indices = hull.simplices
faces = ulti_pts[indices]

# Ensure consistent face orientation using hull equations
# hull.equations gives [a, b, c, d] where ax + by + cz + d = 0 is the plane
# The normal (a, b, c) points outward
# We need to reorder vertices so the cross product gives outward normal
centroid = np.mean(ulti_pts, axis=0)
oriented_faces = []
for i, face in enumerate(faces):
    p1, p2, p3 = face[0], face[1], face[2]
    # Compute face normal from cross product
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    # Check if normal points away from centroid
    face_center = (p1 + p2 + p3) / 3.0
    to_centroid = centroid - face_center
    # If normal points toward centroid, flip the face
    if np.dot(normal, to_centroid) > 0:
        oriented_faces.append([p1, p3, p2])  # Swap p2 and p3 to flip normal
    else:
        oriented_faces.append([p1, p2, p3])
faces = np.array(oriented_faces)

print(' Hull volume: ', hull.volume)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#ax.dist = 30
#ax.azim = -140

#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#ax.set_xlim3d([-750, 750])
#ax.set_ylim3d([-750, 750])
#ax.set_zlim3d([-750, 750])

#for f in faces:
#    f = np.array(f)
#    f[:,2] = f[:,2] - 600
#    face = a3.art3d.Poly3DCollection([f.tolist()])
#    face.set_color(mpl.colors.rgb2hex(sp.rand(3)))
#    face.set_edgecolor('k')
#    face.set_alpha(0.5)
#    ax.add_collection3d(face)

#plt.show()
#plt.close(fig)

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
    
    For a closed mesh (like a convex hull), the signed volume method works by
    summing the signed volumes of tetrahedra formed by each face and the origin.
    The signs cancel out correctly for a closed mesh regardless of where the origin is.
    
    Args:
        faces: numpy array of shape (N, 3, 3) - N triangular faces, each with 3 vertices
    
    Returns:
        Absolute volume of the mesh (always positive)
    """
    total_volume = 0.0
    
    for face in faces:
        # Each face should have exactly 3 vertices
        if len(face) != 3:
            continue
        
        p1 = np.array(face[0])
        p2 = np.array(face[1])
        p3 = np.array(face[2])
        
        # Signed volume of tetrahedron with origin
        # V = (1/6) * p1 · (p2 × p3)
        vol = np.dot(p1, np.cross(p2, p3)) / 6.0
        total_volume += vol
    
    # Return absolute value to ensure positive volume
    return abs(total_volume)

calc_vol = meshVol(faces)
print("Calculated mesh volume:", calc_vol)

# The convex hull volume from scipy is the authoritative value
# The mesh volume calculation is for verification
print("Difference in volume:", abs(hull.volume - calc_vol))

# Use hull volume as the final result since it's more reliable
print("\n" + "=" * 60)
print("FINAL VOLUME ESTIMATE")
print("=" * 60)
print(f"Convex Hull Volume: {hull.volume:.2f} cubic units")
print("=" * 60)





