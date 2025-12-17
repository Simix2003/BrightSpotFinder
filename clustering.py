"""Clustering and blob extraction functions for bright spot detection."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from image_io import USE_OPENCV
from models import Cluster

# Import cv2 if available
if USE_OPENCV:
    import cv2
else:
    cv2 = None


def cluster_bright_points(
    points: List[Tuple[int, int]],
    eps_mm: float,
    min_samples: int,
    pixels_per_mm_x: float,
    pixels_per_mm_y: float,
) -> List[List[Tuple[int, int]]]:
    """Cluster bright points using DBSCAN algorithm with physical distance threshold.
    
    Args:
        points: List of (x, y) coordinates of bright pixels
        eps_mm: Maximum distance in mm between points in the same cluster
        min_samples: Minimum number of points required to form a cluster
        pixels_per_mm_x: Conversion ratio for X axis
        pixels_per_mm_y: Conversion ratio for Y axis
        
    Returns:
        List of clusters, where each cluster is a list of (x, y) coordinates
    """
    if len(points) == 0:
        return []
    
    # Try to import sklearn for DBSCAN
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        # Fallback to simple distance-based clustering if sklearn not available
        return _simple_cluster_points(points, eps_mm, min_samples, pixels_per_mm_x, pixels_per_mm_y)
    
    # Convert points to numpy array
    points_array = np.array(points)
    
    # Convert eps from mm to pixels (use average of x and y ratios for isotropic clustering)
    # For more accurate results, we could use a distance metric that accounts for different ratios
    pixels_per_mm_avg = (pixels_per_mm_x + pixels_per_mm_y) / 2.0
    eps_pixels = eps_mm * pixels_per_mm_avg
    
    # Apply DBSCAN
    clustering = DBSCAN(eps=eps_pixels, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(points_array)
    
    # Group points by cluster label
    # IMPORTANT: Points with label = -1 are noise points (isolated dots that don't form clusters)
    # These are EXCLUDED from the results - we only return actual clusters (neighbors)
    clusters: Dict[int, List[Tuple[int, int]]] = {}
    for idx, label in enumerate(labels):
        if label == -1:  # Noise points (isolated dots, not in any cluster) - SKIP THESE
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(points[idx])
    
    return list(clusters.values())


def extract_blobs_connected_components(
    mask: np.ndarray,
    pixels_per_mm_x: float,
    pixels_per_mm_y: float,
    min_area_mm2: Optional[float] = None,
    min_blob_area_mm2: Optional[float] = None,
    include_points: bool = False,
) -> List[Cluster]:
    """
    Extract blobs from binary mask using connected components.
    
    Args:
        mask: Binary mask (uint8, 0 or 255)
        pixels_per_mm_x: X-axis pixel-to-mm ratio
        pixels_per_mm_y: Y-axis pixel-to-mm ratio
        min_area_mm2: Minimum area threshold (legacy param, same as min_blob_area_mm2)
        min_blob_area_mm2: Minimum blob area in mm² to keep
        include_points: If True, include all pixel coordinates (memory intensive)
    
    Returns:
        List of Cluster objects
    """
    if mask.size == 0:
        return []
    
    # Determine minimum area threshold
    min_area = min_blob_area_mm2 if min_blob_area_mm2 is not None else min_area_mm2
    
    # Use OpenCV if available (much faster)
    if USE_OPENCV:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        clusters: List[Cluster] = []
        for i in range(1, num_labels):  # Skip label 0 (background)
            area_pixels = int(stats[i, cv2.CC_STAT_AREA])
            
            # Convert area from pixels² to mm²
            area_mm2 = area_pixels / (pixels_per_mm_x * pixels_per_mm_y)
            area_cm2 = area_mm2 / 100.0
            
            # Filter by minimum area
            if min_area is not None and area_mm2 < min_area:
                continue
            
            # Get bounding box
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            bounding_box = (x, y, w, h)
            
            # Get centroid
            centroid = (float(centroids[i, 0]), float(centroids[i, 1]))
            
            # Extract points if requested
            points: List[Tuple[int, int]] = []
            if include_points:
                blob_mask = (labels == i)
                y_coords, x_coords = np.where(blob_mask)
                points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
            
            cluster = Cluster(
                cluster_id=i - 1,  # 0-indexed
                points=points,
                area_mm2=area_mm2,
                area_cm2=area_cm2,
                centroid=centroid,
                bounding_box=bounding_box,
            )
            clusters.append(cluster)
        
        return clusters
    else:
        # Fallback: BFS-based connected components
        return _extract_blobs_bfs(mask, pixels_per_mm_x, pixels_per_mm_y, min_area, include_points)


def _extract_blobs_bfs(
    mask: np.ndarray,
    pixels_per_mm_x: float,
    pixels_per_mm_y: float,
    min_area_mm2: Optional[float],
    include_points: bool,
) -> List[Cluster]:
    """BFS-based connected components fallback when OpenCV is not available."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    clusters: List[Cluster] = []
    cluster_id = 0
    
    # Directions for 8-connectivity
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0 and not visited[y, x]:
                # Start BFS for new blob
                queue = [(x, y)]
                visited[y, x] = True
                points: List[Tuple[int, int]] = [(x, y)] if include_points else []
                min_x, max_x = x, x
                min_y, max_y = y, y
                blob_pixels = 1  # Count pixels during BFS
                
                while queue:
                    cx, cy = queue.pop(0)
                    
                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if mask[ny, nx] > 0 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((nx, ny))
                                blob_pixels += 1
                                if include_points:
                                    points.append((nx, ny))
                                min_x = min(min_x, nx)
                                max_x = max(max_x, nx)
                                min_y = min(min_y, ny)
                                max_y = max(max_y, ny)
                
                # Calculate area from pixel count
                area_pixels = blob_pixels
                area_mm2 = area_pixels / (pixels_per_mm_x * pixels_per_mm_y)
                area_cm2 = area_mm2 / 100.0
                
                # Filter by minimum area
                if min_area_mm2 is not None and area_mm2 < min_area_mm2:
                    continue
                
                # Calculate centroid
                if include_points and points:
                    centroid = (float(np.mean([p[0] for p in points])), float(np.mean([p[1] for p in points])))
                else:
                    centroid = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
                
                bounding_box = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
                
                cluster = Cluster(
                    cluster_id=cluster_id,
                    points=points,
                    area_mm2=area_mm2,
                    area_cm2=area_cm2,
                    centroid=centroid,
                    bounding_box=bounding_box,
                )
                clusters.append(cluster)
                cluster_id += 1
    
    return clusters


def _simple_cluster_points(
    points: List[Tuple[int, int]],
    eps_mm: float,
    min_samples: int,
    pixels_per_mm_x: float,
    pixels_per_mm_y: float,
) -> List[List[Tuple[int, int]]]:
    """Simple distance-based clustering fallback when sklearn is not available."""
    if len(points) < min_samples:
        return []
    
    pixels_per_mm_avg = (pixels_per_mm_x + pixels_per_mm_y) / 2.0
    eps_pixels = eps_mm * pixels_per_mm_avg
    eps_pixels_sq = eps_pixels * eps_pixels  # Use squared distance to avoid sqrt
    
    clusters: List[List[Tuple[int, int]]] = []
    unassigned = set(range(len(points)))
    
    while unassigned:
        # Start new cluster with first unassigned point
        seed_idx = unassigned.pop()
        cluster = [points[seed_idx]]
        queue = [seed_idx]
        
        while queue:
            current_idx = queue.pop(0)
            current_point = points[current_idx]
            
            # Find nearby unassigned points
            to_add = []
            for other_idx in list(unassigned):
                other_point = points[other_idx]
                dx = current_point[0] - other_point[0]
                dy = current_point[1] - other_point[1]
                dist_sq = dx * dx + dy * dy
                
                if dist_sq <= eps_pixels_sq:
                    to_add.append(other_idx)
            
            # Add nearby points to cluster
            for idx in to_add:
                unassigned.remove(idx)
                cluster.append(points[idx])
                queue.append(idx)
        
        # Only keep clusters with enough points
        if len(cluster) >= min_samples:
            clusters.append(cluster)
    
    return clusters


def calculate_cluster_area(
    cluster_points: List[Tuple[int, int]],
    pixels_per_mm_x: float,
    pixels_per_mm_y: float,
) -> Tuple[float, float, Tuple[int, int, int, int]]:
    """Calculate cluster area (superficie) in mm² and cm² using convex hull or bounding box.
    
    Args:
        cluster_points: List of (x, y) coordinates in the cluster
        pixels_per_mm_x: Conversion ratio for X axis
        pixels_per_mm_y: Conversion ratio for Y axis
        
    Returns:
        Tuple of (area_mm2, area_cm2, bounding_box) where bounding_box is (x, y, width, height)
    """
    if len(cluster_points) == 0:
        return 0.0, 0.0, (0, 0, 0, 0)
    
    if len(cluster_points) == 1:
        # Single point: area is 1 pixel converted to mm²
        area_pixels = 1.0
        area_mm2 = area_pixels / (pixels_per_mm_x * pixels_per_mm_y)
        area_cm2 = area_mm2 / 100.0
        x, y = cluster_points[0]
        return area_mm2, area_cm2, (x, y, 1, 1)
    
    # Calculate bounding box
    xs = [p[0] for p in cluster_points]
    ys = [p[1] for p in cluster_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width_pixels = max_x - min_x + 1
    height_pixels = max_y - min_y + 1
    
    # Try to use convex hull for more accurate area calculation
    try:
        from scipy.spatial import ConvexHull
        points_array = np.array(cluster_points)
        if len(points_array) >= 3:
            hull = ConvexHull(points_array)
            # Calculate area of convex hull in pixels²
            area_pixels = hull.volume  # For 2D, volume is area
        else:
            # Fallback to bounding box for small clusters
            area_pixels = width_pixels * height_pixels
    except (ImportError, Exception):
        # Fallback to bounding box if scipy not available or error
        area_pixels = width_pixels * height_pixels
    
    # Convert area from pixels² to mm²
    # Area conversion: pixels² -> mm² = pixels² / (pixels_per_mm_x * pixels_per_mm_y)
    area_mm2 = area_pixels / (pixels_per_mm_x * pixels_per_mm_y)
    area_cm2 = area_mm2 / 100.0
    
    bounding_box = (min_x, min_y, width_pixels, height_pixels)
    
    return area_mm2, area_cm2, bounding_box

