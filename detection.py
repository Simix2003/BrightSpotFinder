"""Detection algorithms for bright spot detection."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from typing import List, Optional, Sequence, Tuple

import numpy as np

from clustering import extract_blobs_connected_components
from image_io import USE_OPENCV
from models import Cluster, CropResult

# Import cv2 if available
if USE_OPENCV:
    import cv2
else:
    cv2 = None


def compute_residual_mask(
    crop: np.ndarray,
    blur_ksize: int = 51,
    percentile: float = 99.7,
    min_residual: int = 10,
    morph_ksize: int = 3,
) -> np.ndarray:
    """
    Compute residual mask using background subtraction.
    
    Args:
        crop: Grayscale crop image (uint8 numpy array)
        blur_ksize: Gaussian blur kernel size (must be odd)
        percentile: Percentile for adaptive threshold (0-100)
        min_residual: Minimum residual value floor
        morph_ksize: Morphology kernel size for cleanup (must be odd)
    
    Returns:
        Binary mask (uint8, 0 or 255)
    
    Raises:
        ImportError: If OpenCV is not available
        ValueError: If parameters are invalid
    """
    if not USE_OPENCV:
        raise ImportError(
            "Residual detector requires OpenCV. Install with: pip install opencv-python"
        )
    
    # Ensure kernel sizes are odd
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if morph_ksize % 2 == 0:
        morph_ksize += 1
    
    # Validate parameters
    if blur_ksize < 3:
        raise ValueError(f"blur_ksize must be >= 3, got {blur_ksize}")
    if not (0 <= percentile <= 100):
        raise ValueError(f"percentile must be in range [0, 100], got {percentile}")
    if morph_ksize < 1:
        raise ValueError(f"morph_ksize must be >= 1, got {morph_ksize}")
    
    # Apply Gaussian blur for background
    background = cv2.GaussianBlur(crop, (blur_ksize, blur_ksize), 0)
    
    # Compute residual (crop - background)
    residual = cv2.subtract(crop, background)
    residual = np.clip(residual, 0, 255).astype(np.uint8)
    
    # Calculate threshold from percentile
    threshold = np.percentile(residual, percentile)
    threshold = max(threshold, min_residual)
    
    # Create binary mask
    mask = (residual >= threshold).astype(np.uint8) * 255
    
    # Morphology cleanup: OPEN then CLOSE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def save_debug_artifacts(
    output_dir: Path,
    image_name: str,
    crop: np.ndarray,
    background: np.ndarray,
    residual: np.ndarray,
    mask: np.ndarray,
    blobs: List[Cluster],
) -> None:
    """Save debug images for residual method."""
    if not USE_OPENCV:
        return  # Can't save without OpenCV
    
    debug_dir = output_dir / "debug_residual"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Save residual image (normalized to 0-255)
    if residual.max() > residual.min():
        residual_norm = ((residual - residual.min()) / (residual.max() - residual.min() + 1e-6) * 255).astype(np.uint8)
    else:
        residual_norm = residual.astype(np.uint8)
    cv2.imwrite(str(debug_dir / f"{image_name}_residual.png"), residual_norm)
    
    # Save mask
    cv2.imwrite(str(debug_dir / f"{image_name}_mask.png"), mask)
    
    # Save overlay with blob bounding boxes
    overlay = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for blob in blobs:
        x, y, w, h = blob.bounding_box
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(overlay, (int(blob.centroid[0]), int(blob.centroid[1])), 5, (0, 0, 255), -1)
    cv2.imwrite(str(debug_dir / f"{image_name}_overlay.png"), overlay)


def detect_bright_spots(
    gray_np: np.ndarray,
    crops: Sequence[Tuple[int, int, int, int]],
    threshold: int,
    *,
    max_points: Optional[int] = None,
    module_width_mm: Optional[float] = None,
    module_height_mm: Optional[float] = None,
    bright_spot_area_threshold: Optional[float] = None,
    area_unit: str = "mm",
    dbscan_eps_mm: float = 2.0,
    dbscan_min_samples: int = 3,
    detector: str = "threshold",
    resid_blur_ksize: int = 51,
    resid_percentile: float = 99.7,
    resid_min: int = 10,
    resid_morph_ksize: int = 3,
    min_blob_area_mm2: Optional[float] = None,
    debug_residual: bool = False,
    debug_max: int = 50,
    output_dir: Optional[Path] = None,
    image_name: Optional[str] = None,
    calculate_pixel_to_mm_ratio_func=None,
) -> List[CropResult]:
    """Detect bright spots in image crops with clustering and area filtering.
    
    Args:
        gray_np: Grayscale image as numpy array (uint8)
        crops: Sequence of crop rectangles (x, y, w, h)
        threshold: Brightness threshold (0-255) - used for threshold detector
        max_points: Maximum number of coordinates to extract (None = skip extraction)
        module_width_mm: Module width in millimeters (for physical measurements)
        module_height_mm: Module height in millimeters (for physical measurements)
        bright_spot_area_threshold: Minimum area threshold for bright spots (in mm² or cm²)
        area_unit: Unit for area threshold ("mm" for mm² or "cm" for cm²)
        dbscan_eps_mm: DBSCAN distance threshold in mm (deprecated, kept for compatibility)
        dbscan_min_samples: DBSCAN minimum samples per cluster (deprecated, kept for compatibility)
        detector: Detection method ("threshold" or "residual")
        resid_blur_ksize: Gaussian blur kernel size for residual method
        resid_percentile: Percentile for adaptive threshold in residual method
        resid_min: Minimum residual value floor
        resid_morph_ksize: Morphology kernel size for cleanup in residual method
        min_blob_area_mm2: Minimum blob area in mm² (alternative to bright_spot_area_threshold)
        debug_residual: Enable debug output for residual method
        debug_max: Maximum number of images to generate debug artifacts for
        output_dir: Output directory for debug artifacts
        image_name: Image name for debug artifacts
        calculate_pixel_to_mm_ratio_func: Function to calculate pixel-to-mm ratio
    
    Returns:
        List of CropResult objects with cluster information
    """
    results: List[CropResult] = []
    
    # Validate detector
    if detector not in ("threshold", "residual"):
        raise ValueError(f"Unknown detector: {detector}. Must be 'threshold' or 'residual'")
    
    # Calculate pixel-to-mm ratios if module dimensions provided
    pixels_per_mm_x = 1.0
    pixels_per_mm_y = 1.0
    if module_width_mm is not None and module_height_mm is not None:
        if calculate_pixel_to_mm_ratio_func:
            pixels_per_mm_x, pixels_per_mm_y = calculate_pixel_to_mm_ratio_func(
                crops, module_width_mm, module_height_mm
            )
        else:
            # Fallback calculation
            if crops and len(crops) > 0:
                x, y, w, h = crops[0]
                pixels_per_mm_x = w / module_width_mm if module_width_mm > 0 else 1.0
                pixels_per_mm_y = h / module_height_mm if module_height_mm > 0 else 1.0
    
    # Convert area threshold to mm² if provided
    area_threshold_mm2 = None
    if bright_spot_area_threshold is not None:
        if area_unit == "cm":
            area_threshold_mm2 = bright_spot_area_threshold * 100.0  # cm² to mm²
        else:
            area_threshold_mm2 = bright_spot_area_threshold  # Already in mm²
    
    # For residual method: two-pass approach to calculate global percentile threshold
    global_threshold = None
    if detector == "residual":
        # First pass: collect all residual values from all crops
        all_residuals = []
        for x, y, w, h in crops:
            x2, y2 = x + w, y + h
            crop = gray_np[y:y2, x:x2]
            if USE_OPENCV:
                background = cv2.GaussianBlur(crop, (resid_blur_ksize, resid_blur_ksize), 0)
                residual = cv2.subtract(crop, background)
                residual = np.clip(residual, 0, 255).astype(np.uint8)
                all_residuals.extend(residual.flatten())
            else:
                raise ImportError(
                    "Residual detector requires OpenCV. Install with: pip install opencv-python"
                )
        
        # Calculate global threshold
        if all_residuals:
            global_threshold = float(np.percentile(all_residuals, resid_percentile))
            global_threshold = max(global_threshold, resid_min)
        else:
            global_threshold = resid_min
    
    # Track debug image count
    debug_count = 0
    
    # Main detection loop
    for idx, (x, y, w, h) in enumerate(crops):
        x2, y2 = x + w, y + h
        crop = gray_np[y:y2, x:x2]
        
        # Initialize variables for debug (only used if residual detector)
        background: Optional[np.ndarray] = None
        residual: Optional[np.ndarray] = None
        
        # Generate mask based on detector type
        if detector == "threshold":
            mask = (crop >= threshold).astype(np.uint8) * 255
        elif detector == "residual":
            # Second pass: apply global threshold
            background = cv2.GaussianBlur(crop, (resid_blur_ksize, resid_blur_ksize), 0)
            residual = cv2.subtract(crop, background)
            residual = np.clip(residual, 0, 255).astype(np.uint8)
            mask = (residual >= global_threshold).astype(np.uint8) * 255
            
            # Morphology cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (resid_morph_ksize, resid_morph_ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError(f"Unknown detector: {detector}")
        
        # Calculate bright pixels from mask
        bright_pixels = int(np.count_nonzero(mask))
        total_pixels = mask.size
        ratio = bright_pixels / total_pixels if total_pixels else 0.0
        
        # Extract blobs using connected components (replaces DBSCAN)
        blobs: List[Cluster] = []
        if module_width_mm is not None and module_height_mm is not None:
            blobs = extract_blobs_connected_components(
                mask, pixels_per_mm_x, pixels_per_mm_y,
                min_area_mm2=area_threshold_mm2,
                min_blob_area_mm2=min_blob_area_mm2,
                include_points=(max_points is not None and max_points > 0)
            )
        
        # Convert blobs to Cluster objects and calculate stats
        clusters: List[Cluster] = []
        filtered_clusters: List[Cluster] = []
        max_cluster_area_mm2 = 0.0
        exceeds_threshold = False
        total_bright_spot_area_mm2 = 0.0
        
        for blob in blobs:
            if blob.area_mm2 > max_cluster_area_mm2:
                max_cluster_area_mm2 = blob.area_mm2
            
            if area_threshold_mm2 is not None and blob.area_mm2 >= area_threshold_mm2:
                exceeds_threshold = True
            
            clusters.append(blob)
            
            if area_threshold_mm2 is None or blob.area_mm2 >= area_threshold_mm2:
                filtered_clusters.append(blob)
                total_bright_spot_area_mm2 += blob.area_mm2
        
        # Extract coordinates for display (if needed)
        coords: List[Tuple[int, int]] = []
        if max_points is not None and max_points > 0:
            bright_indices = np.argwhere(mask > 0)
            all_coords = [(int(x + c), int(y + r)) for r, c in bright_indices.tolist()]
            if len(all_coords) > max_points:
                step = max(1, len(all_coords) // max_points)
                coords = all_coords[::step]
            else:
                coords = all_coords
        
        # Debug artifacts (if enabled and under limit)
        # In residual-only mode, save for all crops regardless of bright pixel count
        should_save_debug = (
            debug_residual 
            and debug_count < debug_max 
            and output_dir is not None 
            and image_name is not None 
            and detector == "residual"
        )
        if should_save_debug:
            if background is not None and residual is not None:
                save_debug_artifacts(
                    output_dir, f"{image_name}_crop{idx}",
                    crop, background, residual, mask, blobs
                )
                debug_count += 1
        
        results.append(
            CropResult(
                crop_id=idx,
                rect=(x, y, w, h),
                bright_pixels=bright_pixels,
                total_pixels=total_pixels,
                ratio=ratio,
                bright_coords=coords,
                clusters=clusters if clusters else None,
                filtered_clusters=filtered_clusters if filtered_clusters else None,
                cluster_count=len(clusters),
                filtered_cluster_count=len(filtered_clusters),
                total_bright_spot_area_mm2=total_bright_spot_area_mm2,
                max_cluster_area_mm2=max_cluster_area_mm2,
                exceeds_threshold=exceeds_threshold,
            )
        )
    return results

