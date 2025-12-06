"""
Additional 3D reconstruction methods inspired by multi-view stereo techniques.
Based on methods from "A Comparison and Evaluation of Multi-View Stereo Reconstruction Algorithms"

These methods can complement the existing triangulation approach in volume-test.py
"""

import cv2 as cv
import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import binary_erosion, binary_dilation
import open3d as o3d


def extract_silhouette(image, threshold_method='otsu', debug=False):
    """
    Extract object silhouette from image using background subtraction.
    
    Args:
        image: Grayscale image
        threshold_method: 'otsu', 'adaptive', or 'manual'
        debug: If True, print threshold info
    
    Returns:
        Binary mask where 1 = object, 0 = background
    """
    if threshold_method == 'otsu':
        # Otsu's method automatically finds optimal threshold
        threshold_value, mask = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        if debug:
            print(f"    Otsu threshold: {threshold_value}")
    elif threshold_method == 'adaptive':
        # Adaptive thresholding for varying lighting
        mask = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY_INV, 11, 2)
    elif threshold_method == 'percentile':
        # Use percentile-based threshold (good for consistent backgrounds)
        threshold_value = np.percentile(image, 70)  # Assume background is brighter
        _, mask = cv.threshold(image, threshold_value, 255, cv.THRESH_BINARY_INV)
        if debug:
            print(f"    Percentile threshold: {threshold_value}")
    else:
        # Manual threshold (assume background is lighter, object is darker)
        _, mask = cv.threshold(image, 180, 255, cv.THRESH_BINARY_INV)
    
    # Clean up mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)  # Fill holes
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)   # Remove noise
    
    # Find largest connected component (assume it's the object)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # Get largest component (excluding background which is label 0)
        largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    
    return mask.astype(bool)


def compute_visual_hull_voxels(images, camera_params, voxel_resolution=64):
    """
    Compute visual hull using space carving with voxel grid.
    
    This is a volumetric approach that carves away voxels that don't project
    into the object silhouette in any view.
    
    Args:
        images: List of grayscale images
        camera_params: List of dicts with 'K', 'R', 't' for each view
        voxel_resolution: Number of voxels along each axis
    
    Returns:
        3D binary voxel grid (1 = inside object, 0 = outside)
    """
    # Estimate bounding box from camera positions
    camera_positions = []
    for params in camera_params:
        # Camera position in world coords: C = -R^T * t
        C = -params['R'].T @ params['t']
        camera_positions.append(C.flatten())
    
    camera_positions = np.array(camera_positions)
    
    # Create bounding box centered at origin
    # Assume object is at origin, cameras are around it
    max_dist = np.max(np.linalg.norm(camera_positions, axis=1))
    
    # Use a more generous bounding box (20% of camera distance)
    # This ensures we capture the full object
    bbox_size = max_dist * 0.2
    
    print(f"Camera distance range: {np.min(np.linalg.norm(camera_positions, axis=1)):.2f} - {max_dist:.2f}")
    print(f"Bounding box size: ±{bbox_size:.2f} units")
    
    # Initialize voxel grid
    voxel_grid = np.ones((voxel_resolution, voxel_resolution, voxel_resolution), dtype=bool)
    
    # Create coordinate grid
    x = np.linspace(-bbox_size, bbox_size, voxel_resolution)
    y = np.linspace(-bbox_size, bbox_size, voxel_resolution)
    z = np.linspace(-bbox_size, bbox_size, voxel_resolution)
    
    # Extract silhouettes from all images
    print("Extracting silhouettes...")
    silhouettes = []
    for idx, img in enumerate(images):
        silhouette = extract_silhouette(img, threshold_method='percentile', debug=True)
        silhouettes.append(silhouette)
        # Count foreground pixels
        fg_pixels = np.sum(silhouette)
        total_pixels = silhouette.size
        fg_percentage = (fg_pixels / total_pixels) * 100
        print(f"  View {idx + 1}: {fg_pixels} foreground pixels ({fg_percentage:.1f}%)")
    
    print(f"Carving voxel grid ({voxel_resolution}^3 = {voxel_resolution**3} voxels)...")
    
    # For each view, carve away voxels that don't project into silhouette
    for view_idx, (silhouette, params) in enumerate(zip(silhouettes, camera_params)):
        K = params['K']
        R = params['R']
        t = params['t']
        P = K @ np.hstack([R, t])  # Projection matrix
        
        height, width = silhouette.shape
        carved_count = 0
        
        # Check each voxel
        for i in range(voxel_resolution):
            for j in range(voxel_resolution):
                for k in range(voxel_resolution):
                    if not voxel_grid[i, j, k]:
                        continue  # Already carved
                    
                    # Get 3D world position of voxel center
                    point_3d = np.array([x[i], y[j], z[k], 1.0])
                    
                    # Project to image
                    point_2d_h = P @ point_3d
                    
                    if point_2d_h[2] <= 0:
                        # Behind camera
                        voxel_grid[i, j, k] = False
                        carved_count += 1
                        continue
                    
                    # Convert to pixel coordinates
                    u = int(point_2d_h[0] / point_2d_h[2])
                    v = int(point_2d_h[1] / point_2d_h[2])
                    
                    # Check if projection is outside silhouette
                    if u < 0 or u >= width or v < 0 or v >= height:
                        voxel_grid[i, j, k] = False
                        carved_count += 1
                    elif not silhouette[v, u]:
                        voxel_grid[i, j, k] = False
                        carved_count += 1
        
        print(f"  View {view_idx + 1}: Carved {carved_count} voxels")
    
    remaining_voxels = np.sum(voxel_grid)
    print(f"Visual hull complete: {remaining_voxels} voxels remaining")
    
    return voxel_grid, (x, y, z)


def voxel_grid_to_point_cloud(voxel_grid, coordinates):
    """
    Convert binary voxel grid to point cloud.
    
    Args:
        voxel_grid: 3D binary array
        coordinates: Tuple of (x, y, z) coordinate arrays
    
    Returns:
        Nx3 array of 3D points
    """
    x, y, z = coordinates
    points = []
    
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[2]):
                if voxel_grid[i, j, k]:
                    points.append([x[i], y[j], z[k]])
    
    return np.array(points)


def fuse_depth_maps_weighted(depth_maps, confidence_maps, camera_params, 
                             output_resolution=(480, 640)):
    """
    Fuse multiple depth maps using weighted averaging based on confidence.
    
    This creates a denser, more accurate depth map by combining information
    from multiple views.
    
    Args:
        depth_maps: List of depth maps (one per view)
        confidence_maps: List of confidence maps (higher = more reliable)
        camera_params: List of camera parameter dicts
        output_resolution: (height, width) of output depth map
    
    Returns:
        Fused depth map in reference view coordinates
    """
    height, width = output_resolution
    
    # Initialize accumulator
    fused_depth = np.zeros((height, width), dtype=np.float32)
    total_confidence = np.zeros((height, width), dtype=np.float32)
    
    # Use first view as reference
    ref_params = camera_params[0]
    ref_K = ref_params['K']
    ref_R = ref_params['R']
    ref_t = ref_params['t']
    
    # For each source view
    for view_idx, (depth_map, confidence_map, params) in enumerate(
            zip(depth_maps, confidence_maps, camera_params)):
        
        src_K = params['K']
        src_R = params['R']
        src_t = params['t']
        
        # For each pixel in source depth map
        src_height, src_width = depth_map.shape
        
        for v_src in range(src_height):
            for u_src in range(src_width):
                depth = depth_map[v_src, u_src]
                confidence = confidence_map[v_src, u_src]
                
                if depth <= 0 or confidence <= 0:
                    continue
                
                # Backproject to 3D in source camera coordinates
                x_src = (u_src - src_K[0, 2]) * depth / src_K[0, 0]
                y_src = (v_src - src_K[1, 2]) * depth / src_K[1, 1]
                z_src = depth
                
                # Transform to world coordinates
                point_cam = np.array([[x_src], [y_src], [z_src]])
                point_world = src_R.T @ (point_cam - src_t)
                
                # Transform to reference camera coordinates
                point_ref_cam = ref_R @ point_world + ref_t
                
                # Project to reference image
                x_ref = point_ref_cam[0, 0]
                y_ref = point_ref_cam[1, 0]
                z_ref = point_ref_cam[2, 0]
                
                if z_ref <= 0:
                    continue
                
                u_ref = int(ref_K[0, 0] * x_ref / z_ref + ref_K[0, 2])
                v_ref = int(ref_K[1, 1] * y_ref / z_ref + ref_K[1, 2])
                
                # Check bounds
                if 0 <= u_ref < width and 0 <= v_ref < height:
                    # Accumulate weighted depth
                    fused_depth[v_ref, u_ref] += z_ref * confidence
                    total_confidence[v_ref, u_ref] += confidence
        
        print(f"  Fused depth map from view {view_idx + 1}")
    
    # Normalize by total confidence
    valid_mask = total_confidence > 0
    fused_depth[valid_mask] /= total_confidence[valid_mask]
    
    return fused_depth


def compute_depth_confidence(depth_map, method='gradient'):
    """
    Compute confidence map for depth values.
    
    Lower gradient = higher confidence (smoother surface)
    
    Args:
        depth_map: 2D depth map
        method: 'gradient' or 'uniform'
    
    Returns:
        Confidence map (0-1, higher = more confident)
    """
    if method == 'uniform':
        # Uniform confidence
        return np.ones_like(depth_map)
    
    elif method == 'gradient':
        # Confidence based on depth gradient (smooth areas = high confidence)
        grad_x = cv.Sobel(depth_map, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(depth_map, cv.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and invert (low gradient = high confidence)
        max_grad = np.percentile(gradient_magnitude, 95)  # Ignore outliers
        confidence = 1.0 - np.clip(gradient_magnitude / max_grad, 0, 1)
        
        return confidence


def patch_based_stereo_matching(img1, img2, K1, R1, t1, K2, R2, t2, 
                                patch_size=11, search_range=50):
    """
    Compute dense depth map using patch-based stereo matching.
    
    This is more robust than simple block matching as it uses photometric
    consistency across multiple views.
    
    Args:
        img1, img2: Grayscale images
        K1, R1, t1: Camera parameters for view 1
        K2, R2, t2: Camera parameters for view 2
        patch_size: Size of matching patch (odd number)
        search_range: Maximum disparity to search
    
    Returns:
        Depth map for view 1
    """
    height, width = img1.shape
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    half_patch = patch_size // 2
    
    # Compute fundamental matrix for epipolar constraint
    # F = K2^-T * [t]_x * R * K1^-1
    # where R = R2 * R1^T, t = t2 - R * t1
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    
    # Skew-symmetric matrix for cross product
    t_cross = np.array([[0, -t_rel[2, 0], t_rel[1, 0]],
                       [t_rel[2, 0], 0, -t_rel[0, 0]],
                       [-t_rel[1, 0], t_rel[0, 0], 0]])
    
    F = np.linalg.inv(K2).T @ t_cross @ R_rel @ np.linalg.inv(K1)
    
    print(f"Computing patch-based stereo (patch size: {patch_size}x{patch_size})...")
    
    # For each pixel in img1
    for v1 in range(half_patch, height - half_patch, 2):  # Subsample for speed
        for u1 in range(half_patch, width - half_patch, 2):
            # Extract patch from img1
            patch1 = img1[v1-half_patch:v1+half_patch+1, 
                         u1-half_patch:u1+half_patch+1]
            
            # Compute epipolar line in img2
            point1_h = np.array([u1, v1, 1.0])
            epipolar_line = F @ point1_h  # ax + by + c = 0
            
            # Search along epipolar line
            best_match_score = float('inf')
            best_u2 = -1
            best_v2 = -1
            
            for u2 in range(max(half_patch, u1 - search_range), 
                          min(width - half_patch, u1 + search_range)):
                # Compute v2 from epipolar constraint: v2 = -(ax + c) / b
                if abs(epipolar_line[1]) > 1e-6:
                    v2 = int(-(epipolar_line[0] * u2 + epipolar_line[2]) / epipolar_line[1])
                else:
                    v2 = v1  # Approximately horizontal epipolar line
                
                if v2 < half_patch or v2 >= height - half_patch:
                    continue
                
                # Extract patch from img2
                patch2 = img2[v2-half_patch:v2+half_patch+1,
                             u2-half_patch:u2+half_patch+1]
                
                # Compute matching cost (SSD)
                if patch1.shape == patch2.shape:
                    ssd = np.sum((patch1.astype(float) - patch2.astype(float)) ** 2)
                    
                    if ssd < best_match_score:
                        best_match_score = ssd
                        best_u2 = u2
                        best_v2 = v2
            
            # Triangulate to get depth
            if best_u2 >= 0:
                # Simple triangulation using disparity
                disparity = abs(u1 - best_u2)
                if disparity > 0:
                    # Depth = baseline * focal_length / disparity
                    baseline = np.linalg.norm(t_rel)
                    focal_length = K1[0, 0]
                    depth = baseline * focal_length / disparity
                    depth_map[v1, u1] = depth
    
    # Interpolate to fill gaps
    # (In production, use more sophisticated hole filling)
    
    return depth_map


def surface_growing_from_seeds(seed_points, images, camera_params, 
                               max_iterations=100, growth_threshold=0.8):
    """
    Grow surface from high-confidence seed points.
    
    This expands the reconstruction by finding neighboring points that are
    photometrically consistent across views.
    
    Args:
        seed_points: Nx3 array of initial 3D points
        images: List of images
        camera_params: List of camera parameter dicts
        max_iterations: Maximum growth iterations
        growth_threshold: Photometric consistency threshold (0-1)
    
    Returns:
        Expanded point cloud
    """
    # Convert seed points to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(seed_points)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    
    # Build KD-tree for neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    print(f"Growing surface from {len(seed_points)} seed points...")
    
    # Iteratively grow surface
    for iteration in range(max_iterations):
        new_points = []
        
        # For each point, try to grow in normal direction
        for i in range(len(pcd.points)):
            point = np.asarray(pcd.points[i])
            normal = np.asarray(pcd.normals[i])
            
            # Try candidate points along normal
            for step_size in [0.01, 0.02, 0.05]:
                candidate = point + normal * step_size
                
                # Check photometric consistency across views
                consistency = check_photometric_consistency(
                    candidate, images, camera_params)
                
                if consistency > growth_threshold:
                    new_points.append(candidate)
                    break
        
        if len(new_points) == 0:
            print(f"  Surface growth converged at iteration {iteration}")
            break
        
        # Add new points to point cloud
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(np.array(new_points))
        pcd += new_pcd
        
        # Re-estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        
        print(f"  Iteration {iteration + 1}: Added {len(new_points)} points (total: {len(pcd.points)})")
    
    return np.asarray(pcd.points)


def check_photometric_consistency(point_3d, images, camera_params, 
                                  patch_size=5, threshold=0.8):
    """
    Check if a 3D point is photometrically consistent across multiple views.
    
    Args:
        point_3d: 3D point to check
        images: List of images
        camera_params: List of camera parameter dicts
        patch_size: Size of patch to compare
        threshold: Consistency threshold (0-1)
    
    Returns:
        Consistency score (0-1, higher = more consistent)
    """
    half_patch = patch_size // 2
    patches = []
    
    # Project point to each view and extract patch
    for img, params in zip(images, camera_params):
        K = params['K']
        R = params['R']
        t = params['t']
        
        # Transform to camera coordinates
        point_cam = R @ point_3d.reshape(3, 1) + t
        
        if point_cam[2, 0] <= 0:
            continue  # Behind camera
        
        # Project to image
        u = int(K[0, 0] * point_cam[0, 0] / point_cam[2, 0] + K[0, 2])
        v = int(K[1, 1] * point_cam[1, 0] / point_cam[2, 0] + K[1, 2])
        
        # Check bounds
        height, width = img.shape
        if (u < half_patch or u >= width - half_patch or 
            v < half_patch or v >= height - half_patch):
            continue
        
        # Extract patch
        patch = img[v-half_patch:v+half_patch+1, u-half_patch:u+half_patch+1]
        patches.append(patch.flatten())
    
    if len(patches) < 2:
        return 0.0  # Not visible in enough views
    
    # Compute normalized cross-correlation between all patch pairs
    correlations = []
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            # Normalize patches
            p1 = patches[i] - np.mean(patches[i])
            p2 = patches[j] - np.mean(patches[j])
            
            # Compute correlation
            if np.std(p1) > 0 and np.std(p2) > 0:
                corr = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
                correlations.append(corr)
    
    if len(correlations) == 0:
        return 0.0
    
    # Return average correlation
    return np.mean(correlations)


if __name__ == "__main__":
    print("Multi-View Stereo Reconstruction Methods")
    print("=" * 60)
    print("This module provides additional reconstruction methods:")
    print("  1. Visual Hull (Space Carving)")
    print("  2. Depth Map Fusion")
    print("  3. Patch-Based Stereo Matching")
    print("  4. Surface Growing")
    print()
    print("Import these functions into volume-test.py to use them.")
