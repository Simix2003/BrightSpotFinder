"""Detection algorithms for bright spot detection."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from clustering import extract_blobs_connected_components
from image_io import USE_OPENCV, iter_images, load_image_optimized
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


def build_reference_residual_model(
    good_images_dir: Path,
    crops: Sequence[Tuple[int, int, int, int]],
    resid_blur_ksize: int = 51,
    debug: bool = False,
    output_dir: Optional[Path] = None,
    max_good_images: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Build reference residual model from GOOD images.
    
    For each crop, computes residual for all GOOD images and builds a median
    residual image (pixel-wise median across all GOOD residuals).
    
    Args:
        good_images_dir: Directory containing GOOD reference images
        crops: Sequence of crop rectangles (x, y, w, h)
        resid_blur_ksize: Gaussian blur kernel size for residual computation
    
    Returns:
        Dictionary mapping crop_id -> median_residual_image (uint8 numpy array)
    
    Raises:
        ImportError: If OpenCV is not available
        FileNotFoundError: If no GOOD images found
    """
    if not USE_OPENCV:
        raise ImportError(
            "Reference residual detector requires OpenCV. Install with: pip install opencv-python"
        )
    
    # Collect all GOOD image paths
    good_image_paths = list(iter_images(good_images_dir))
    if not good_image_paths:
        raise FileNotFoundError(f"No images found in GOOD images directory: {good_images_dir}")
    
    # Limit to max_good_images if specified (for debug mode)
    if max_good_images is not None and max_good_images > 0:
        good_image_paths = good_image_paths[:max_good_images]
        print(f"[DEBUG] Limiting to {max_good_images} GOOD image(s) for debug mode", file=sys.stderr)
    
    print(f"Building reference residual model from {len(good_image_paths)} GOOD image(s)...", file=sys.stderr)
    
    # Ensure blur kernel size is odd
    if resid_blur_ksize % 2 == 0:
        resid_blur_ksize += 1
    
    # Dictionary to store residuals for each crop
    # Structure: {crop_id: [list of residual arrays from all GOOD images]}
    crop_residuals: Dict[int, List[np.ndarray]] = {}
    
    # Process each GOOD image
    for img_idx, good_img_path in enumerate(good_image_paths):
        if debug:
            print(f"  Processing GOOD image {img_idx + 1}/{len(good_image_paths)}: {good_img_path.name}", file=sys.stderr)
        
        # Load GOOD image as grayscale numpy array
        gray_np, _ = load_image_optimized(good_img_path, as_grayscale=True)
        
        # Process each crop
        for crop_id, (x, y, w, h) in enumerate(crops):
            x2, y2 = x + w, y + h
            crop = gray_np[y:y2, x:x2]
            
            # Compute residual for this crop
            background = cv2.GaussianBlur(crop, (resid_blur_ksize, resid_blur_ksize), 0)
            residual = cv2.subtract(crop, background)
            residual = np.clip(residual, 0, 255).astype(np.uint8)
            
            # Store residual for this crop
            if crop_id not in crop_residuals:
                crop_residuals[crop_id] = []
            crop_residuals[crop_id].append(residual)
            
            if debug and img_idx == 0 and output_dir is not None:
                # Save first GOOD image residual as example
                debug_dir = output_dir / "debug_reference"
                debug_dir.mkdir(parents=True, exist_ok=True)
                if residual.max() > residual.min():
                    residual_norm = ((residual - residual.min()) / (residual.max() - residual.min() + 1e-6) * 255).astype(np.uint8)
                else:
                    residual_norm = residual.astype(np.uint8)
                good_example_path = debug_dir / f"GOOD_example_crop{crop_id}_residual.png"
                success = cv2.imwrite(str(good_example_path), residual_norm)
                if success:
                    print(f"  [DEBUG] ✓ Saved GOOD image residual example: {good_example_path.name}", file=sys.stderr)
                else:
                    print(f"  [DEBUG] ✗ Failed to save GOOD image residual example", file=sys.stderr)
    
    # Build median residual for each crop
    reference_model: Dict[int, np.ndarray] = {}
    for crop_id, residual_list in crop_residuals.items():
        if not residual_list:
            # Should not happen, but handle gracefully
            continue
        
        # Check that all residuals have the same shape
        shapes = [r.shape for r in residual_list]
        if len(set(shapes)) > 1:
            print(f"  [DEBUG] WARNING: Crop {crop_id} has residuals with different shapes: {shapes}", file=sys.stderr)
            # Use the first residual's shape as reference
            target_shape = residual_list[0].shape
            # Resize all residuals to match the first one
            print(f"  [DEBUG] Resizing all residuals to target shape: {target_shape}", file=sys.stderr)
            resized_residuals = []
            for idx, residual in enumerate(residual_list):
                if residual.shape != target_shape:
                    if debug:
                        print(f"    [DEBUG] Resizing residual {idx} from {residual.shape} to {target_shape}", file=sys.stderr)
                    resized = cv2.resize(residual, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                    resized_residuals.append(resized)
                else:
                    resized_residuals.append(residual)
            residual_list = resized_residuals
        
        # Stack all residuals for this crop
        # All residuals should have same shape (same crop size)
        residual_stack = np.stack(residual_list, axis=0)  # Shape: (num_images, height, width)
        
        # Compute pixel-wise median
        median_residual = np.median(residual_stack, axis=0).astype(np.uint8)
        
        reference_model[crop_id] = median_residual
        
        if debug:
            print(f"  Crop {crop_id}: median residual stats - min={median_residual.min()}, max={median_residual.max()}, mean={median_residual.mean():.2f}", file=sys.stderr)
            
            if output_dir is not None:
                # Save reference model (median residual)
                debug_dir = output_dir / "debug_reference"
                debug_dir.mkdir(parents=True, exist_ok=True)
                if median_residual.max() > median_residual.min():
                    ref_norm = ((median_residual - median_residual.min()) / (median_residual.max() - median_residual.min() + 1e-6) * 255).astype(np.uint8)
                else:
                    ref_norm = median_residual.astype(np.uint8)
                filepath = debug_dir / f"REFERENCE_MODEL_crop{crop_id}_median_residual.png"
                success = cv2.imwrite(str(filepath), ref_norm)
                if success:
                    print(f"  [DEBUG] ✓ Saved REFERENCE MODEL (median of all GOOD images) crop {crop_id}: {filepath.name}", file=sys.stderr)
                else:
                    print(f"  [DEBUG] ✗ ERROR: Failed to save reference model crop {crop_id} to {filepath}", file=sys.stderr)
                if crop_id == 0:
                    print(f"  [DEBUG] Reference model images saved to: {debug_dir}", file=sys.stderr)
    
    print(f"Reference residual model built successfully for {len(reference_model)} crop(s).", file=sys.stderr)
    return reference_model


def save_reference_debug_artifacts(
    output_dir: Path,
    image_name: str,
    crop: np.ndarray,
    background: Optional[np.ndarray],
    residual_test: np.ndarray,
    reference_residual: np.ndarray,
    diff: np.ndarray,
    mask: np.ndarray,
    blobs: List[Cluster],
    threshold: float,
) -> None:
    """Save debug images for reference-residual method."""
    print(f"\n[DEBUG] save_reference_debug_artifacts CALLED", file=sys.stderr)
    print(f"  [DEBUG] image_name={image_name}", file=sys.stderr)
    print(f"  [DEBUG] output_dir={output_dir}", file=sys.stderr)
    print(f"  [DEBUG] crop shape={crop.shape}, dtype={crop.dtype}, min={crop.min()}, max={crop.max()}", file=sys.stderr)
    print(f"  [DEBUG] residual_test shape={residual_test.shape}, dtype={residual_test.dtype}, min={residual_test.min()}, max={residual_test.max()}", file=sys.stderr)
    print(f"  [DEBUG] reference_residual shape={reference_residual.shape}, dtype={reference_residual.dtype}, min={reference_residual.min()}, max={reference_residual.max()}", file=sys.stderr)
    print(f"  [DEBUG] diff shape={diff.shape}, dtype={diff.dtype}, min={diff.min()}, max={diff.max()}", file=sys.stderr)
    print(f"  [DEBUG] mask shape={mask.shape}, dtype={mask.dtype}, min={mask.min()}, max={mask.max()}", file=sys.stderr)
    print(f"  [DEBUG] blobs count={len(blobs)}", file=sys.stderr)
    print(f"  [DEBUG] USE_OPENCV={USE_OPENCV}", file=sys.stderr)
    
    if not USE_OPENCV:
        print(f"  [DEBUG] ERROR: OpenCV not available, returning early", file=sys.stderr)
        return  # Can't save without OpenCV
    
    debug_dir = output_dir / "debug_reference"
    print(f"  [DEBUG] Creating debug_dir: {debug_dir}", file=sys.stderr)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify folder was created
    if not debug_dir.exists():
        print(f"  [DEBUG] ERROR: Debug folder was not created at {debug_dir}", file=sys.stderr)
        return
    else:
        print(f"  [DEBUG] Debug folder exists: {debug_dir}", file=sys.stderr)
        print(f"  [DEBUG] Debug folder is writable: {debug_dir.is_dir()}", file=sys.stderr)
    
    # Save test residual (normalized)
    print(f"  [DEBUG] Processing test_residual...", file=sys.stderr)
    if residual_test.max() > residual_test.min():
        test_res_norm = ((residual_test - residual_test.min()) / (residual_test.max() - residual_test.min() + 1e-6) * 255).astype(np.uint8)
    else:
        test_res_norm = residual_test.astype(np.uint8)
    test_res_path = debug_dir / f"{image_name}_test_residual.png"
    print(f"  [DEBUG] Saving test_residual to: {test_res_path}", file=sys.stderr)
    success1 = cv2.imwrite(str(test_res_path), test_res_norm)
    print(f"  [DEBUG] test_residual save result: {success1}, file exists: {test_res_path.exists()}", file=sys.stderr)
    
    # Save reference residual (normalized)
    print(f"  [DEBUG] Processing reference_residual...", file=sys.stderr)
    if reference_residual.max() > reference_residual.min():
        ref_res_norm = ((reference_residual - reference_residual.min()) / (reference_residual.max() - reference_residual.min() + 1e-6) * 255).astype(np.uint8)
    else:
        ref_res_norm = reference_residual.astype(np.uint8)
    ref_res_path = debug_dir / f"{image_name}_reference_residual.png"
    print(f"  [DEBUG] Saving reference_residual to: {ref_res_path}", file=sys.stderr)
    success2 = cv2.imwrite(str(ref_res_path), ref_res_norm)
    print(f"  [DEBUG] reference_residual save result: {success2}, file exists: {ref_res_path.exists()}", file=sys.stderr)
    
    # Save difference image (normalized)
    print(f"  [DEBUG] Processing diff...", file=sys.stderr)
    if diff.max() > diff.min():
        diff_norm = ((diff - diff.min()) / (diff.max() - diff.min() + 1e-6) * 255).astype(np.uint8)
    else:
        diff_norm = diff.astype(np.uint8)
    diff_path = debug_dir / f"{image_name}_difference.png"
    print(f"  [DEBUG] Saving diff to: {diff_path}", file=sys.stderr)
    success3 = cv2.imwrite(str(diff_path), diff_norm)
    print(f"  [DEBUG] diff save result: {success3}, file exists: {diff_path.exists()}", file=sys.stderr)
    
    # Save thresholded mask
    print(f"  [DEBUG] Processing mask...", file=sys.stderr)
    mask_path = debug_dir / f"{image_name}_mask.png"
    print(f"  [DEBUG] Saving mask to: {mask_path}", file=sys.stderr)
    success4 = cv2.imwrite(str(mask_path), mask)
    print(f"  [DEBUG] mask save result: {success4}, file exists: {mask_path.exists()}", file=sys.stderr)
    
    # Save overlay with blob bounding boxes - STEP 5: BLOB DETECTION
    print(f"  [DEBUG] STEP 5: Processing BLOB DETECTION (found {len(blobs)} blob(s))...", file=sys.stderr)
    overlay = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    print(f"  [DEBUG] overlay shape after conversion: {overlay.shape}", file=sys.stderr)
    for blob in blobs:
        x, y, w, h = blob.bounding_box
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(overlay, (int(blob.centroid[0]), int(blob.centroid[1])), 5, (0, 0, 255), -1)
    overlay_path = debug_dir / f"{image_name}_BLOBS.png"
    print(f"  [DEBUG] Saving BLOBS overlay to: {overlay_path.name}", file=sys.stderr)
    success5 = cv2.imwrite(str(overlay_path), overlay)
    print(f"  [DEBUG] {'✓' if success5 else '✗'} BLOBS overlay saved: {success5}, file exists: {overlay_path.exists()}", file=sys.stderr)
    
    # Save side-by-side comparison
    print(f"  [DEBUG] Creating side-by-side COMPARISON...", file=sys.stderr)
    comparison = np.hstack([
        cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(test_res_norm, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(ref_res_norm, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(diff_norm, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
    ])
    comparison_path = debug_dir / f"{image_name}_COMPARISON.png"
    print(f"  [DEBUG] Saving COMPARISON to: {comparison_path.name}", file=sys.stderr)
    success6 = cv2.imwrite(str(comparison_path), comparison)
    print(f"  [DEBUG] {'✓' if success6 else '✗'} COMPARISON saved: {success6}, file exists: {comparison_path.exists()}", file=sys.stderr)
    
    print(f"\n  [DEBUG] {'='*70}", file=sys.stderr)
    print(f"  [DEBUG] SUMMARY - Save results:", file=sys.stderr)
    print(f"    [DEBUG] {'✓' if success1 else '✗'} TEST residual: {success1}", file=sys.stderr)
    print(f"    [DEBUG] {'✓' if success2 else '✗'} REFERENCE residual: {success2}", file=sys.stderr)
    print(f"    [DEBUG] {'✓' if success3 else '✗'} DIFFERENCE: {success3}", file=sys.stderr)
    print(f"    [DEBUG] {'✓' if success4 else '✗'} MASK: {success4}", file=sys.stderr)
    print(f"    [DEBUG] {'✓' if success5 else '✗'} BLOBS: {success5} ({len(blobs)} blob(s) found)", file=sys.stderr)
    print(f"    [DEBUG] {'✓' if success6 else '✗'} COMPARISON: {success6}", file=sys.stderr)
    if not all([success1, success2, success3, success4, success5, success6]):
        print(f"  [DEBUG] ⚠ WARNING: Some debug images failed to save!", file=sys.stderr)
    else:
        print(f"  [DEBUG] ✓ SUCCESS: All debug images saved for {image_name}", file=sys.stderr)
    print(f"  [DEBUG] All files saved to: {debug_dir}", file=sys.stderr)
    print(f"  [DEBUG] {'='*70}\n", file=sys.stderr)


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
    good_images_dir: Optional[Path] = None,
    reference_model: Optional[Dict[int, np.ndarray]] = None,
    reference_diff_percentile: float = 99.5,
    reference_diff_min: int = 5,
    debug_reference: bool = False,
    debug_reference_max: int = 50,
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
        detector: Detection method ("threshold", "residual", or "reference-residual")
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
        good_images_dir: Directory containing GOOD reference images (for reference-residual detector)
        reference_model: Pre-built reference model (optional, will build from good_images_dir if not provided)
        reference_diff_percentile: Percentile for thresholding difference image (0-100)
        reference_diff_min: Minimum difference threshold floor
    
    Returns:
        List of CropResult objects with cluster information
    """
    if debug_reference:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"[DEBUG] detect_bright_spots CALLED", file=sys.stderr)
        print(f"  [DEBUG] detector={detector}", file=sys.stderr)
        print(f"  [DEBUG] debug_reference={debug_reference}", file=sys.stderr)
        print(f"  [DEBUG] output_dir={output_dir}", file=sys.stderr)
        print(f"  [DEBUG] image_name={image_name}", file=sys.stderr)
        print(f"  [DEBUG] good_images_dir={good_images_dir}", file=sys.stderr)
        print(f"  [DEBUG] reference_model is None={reference_model is None}", file=sys.stderr)
        if reference_model is not None:
            print(f"  [DEBUG] reference_model has {len(reference_model)} crop(s): {list(reference_model.keys())}", file=sys.stderr)
        print(f"  [DEBUG] gray_np shape={gray_np.shape}, dtype={gray_np.dtype}", file=sys.stderr)
        print(f"  [DEBUG] number of crops={len(crops)}", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
    
    results: List[CropResult] = []
    
    # Validate detector
    if detector not in ("threshold", "residual", "reference-residual"):
        raise ValueError(f"Unknown detector: {detector}. Must be 'threshold', 'residual', or 'reference-residual'")
    
    # Build reference model if needed
    if detector == "reference-residual":
        print(f"\n[DEBUG] detect_bright_spots: reference-residual detector", file=sys.stderr)
        print(f"  [DEBUG] debug_reference={debug_reference}", file=sys.stderr)
        print(f"  [DEBUG] output_dir={output_dir}", file=sys.stderr)
        print(f"  [DEBUG] image_name={image_name}", file=sys.stderr)
        print(f"  [DEBUG] number of crops={len(crops)}", file=sys.stderr)
        
        # Create debug folder early if debug is enabled
        if debug_reference and output_dir is not None:
            debug_dir = output_dir / "debug_reference"
            print(f"  [DEBUG] Creating debug folder: {debug_dir}", file=sys.stderr)
            debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [DEBUG] Debug folder created: {debug_dir}, exists: {debug_dir.exists()}", file=sys.stderr)
        
        if reference_model is None:
            print(f"  [DEBUG] reference_model is None, building from good_images_dir", file=sys.stderr)
            if good_images_dir is None:
                raise ValueError("good_images_dir is required for reference-residual detector when reference_model is not provided")
            if not USE_OPENCV:
                raise ImportError(
                    "Reference residual detector requires OpenCV. Install with: pip install opencv-python"
                )
            print(f"  [DEBUG] Calling build_reference_residual_model...", file=sys.stderr)
            # In debug mode, only use 1 GOOD image
            max_good_images = 1 if debug_reference else None
            reference_model = build_reference_residual_model(
                good_images_dir, crops, resid_blur_ksize,
                debug=debug_reference, output_dir=output_dir,
                max_good_images=max_good_images
            )
            print(f"  [DEBUG] Reference model built with {len(reference_model)} crop(s): {list(reference_model.keys())}", file=sys.stderr)
            print(f"  [DEBUG] Processing {len(crops)} crop(s) for detection", file=sys.stderr)
        else:
            print(f"  [DEBUG] Using existing reference_model with {len(reference_model)} crop(s): {list(reference_model.keys())}", file=sys.stderr)
    
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
    
    # For reference-residual method: collect all difference values for global threshold
    reference_diff_threshold = None
    if detector == "reference-residual":
        all_differences = []
        for idx, (x, y, w, h) in enumerate(crops):
            x2, y2 = x + w, y + h
            crop = gray_np[y:y2, x:x2]
            if USE_OPENCV:
                # Compute residual for test image
                background = cv2.GaussianBlur(crop, (resid_blur_ksize, resid_blur_ksize), 0)
                residual_test = cv2.subtract(crop, background)
                residual_test = np.clip(residual_test, 0, 255).astype(np.uint8)
                
                # Get reference residual for this crop
                if idx not in reference_model:
                    # Skip if reference model doesn't have this crop (should not happen)
                    continue
                reference_residual = reference_model[idx]
                
                # Compute difference
                diff = np.abs(residual_test.astype(np.int16) - reference_residual.astype(np.int16))
                diff = np.clip(diff, 0, 255).astype(np.uint8)
                all_differences.extend(diff.flatten())
            else:
                raise ImportError(
                    "Reference residual detector requires OpenCV. Install with: pip install opencv-python"
                )
        
        # Calculate global difference threshold
        if all_differences:
            reference_diff_threshold = float(np.percentile(all_differences, reference_diff_percentile))
            reference_diff_threshold = max(reference_diff_threshold, reference_diff_min)
            if debug_reference:
                print(f"  Global difference threshold: {reference_diff_threshold:.2f} (percentile {reference_diff_percentile}%, min={reference_diff_min})", file=sys.stderr)
                print(f"  Difference stats - min={min(all_differences)}, max={max(all_differences)}, mean={sum(all_differences)/len(all_differences):.2f}", file=sys.stderr)
        else:
            reference_diff_threshold = reference_diff_min
            if debug_reference:
                print(f"  Warning: No differences computed, using minimum threshold: {reference_diff_threshold}", file=sys.stderr)
    
    # Track debug image count
    debug_count = 0
    debug_reference_count = 0
    
    # Main detection loop
    for idx, (x, y, w, h) in enumerate(crops):
        x2, y2 = x + w, y + h
        crop = gray_np[y:y2, x:x2]
        
        # Initialize variables for debug (only used if residual detector)
        background: Optional[np.ndarray] = None
        residual: Optional[np.ndarray] = None
        # Initialize variables for reference-residual debug
        diff: Optional[np.ndarray] = None
        reference_residual: Optional[np.ndarray] = None
        
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
        elif detector == "reference-residual":
            if debug_reference:
                print(f"  [DEBUG] Processing crop {idx} with reference-residual method", file=sys.stderr)
                print(f"    [DEBUG] crop shape={crop.shape}, dtype={crop.dtype}", file=sys.stderr)
            
            # Compute residual for test image
            if debug_reference:
                print(f"    [DEBUG] Computing background blur...", file=sys.stderr)
            background = cv2.GaussianBlur(crop, (resid_blur_ksize, resid_blur_ksize), 0)
            if debug_reference:
                print(f"    [DEBUG] background shape={background.shape}, dtype={background.dtype}", file=sys.stderr)
            
            if debug_reference:
                print(f"    [DEBUG] Computing residual (crop - background)...", file=sys.stderr)
            residual = cv2.subtract(crop, background)
            residual = np.clip(residual, 0, 255).astype(np.uint8)
            if debug_reference:
                print(f"    [DEBUG] residual shape={residual.shape}, dtype={residual.dtype}, min={residual.min()}, max={residual.max()}", file=sys.stderr)
            
            # Get reference residual for this crop
            if debug_reference:
                print(f"    [DEBUG] Checking if idx {idx} in reference_model...", file=sys.stderr)
                print(f"    [DEBUG] reference_model keys: {list(reference_model.keys()) if reference_model else 'None'}", file=sys.stderr)
            
            if idx not in reference_model:
                if debug_reference:
                    print(f"    [DEBUG] WARNING: idx {idx} NOT in reference_model!", file=sys.stderr)
                # If reference model doesn't have this crop, create empty mask
                mask = np.zeros_like(crop, dtype=np.uint8)
                diff = None
                reference_residual = None
                if debug_reference:
                    print(f"    [DEBUG] Set diff=None, reference_residual=None, created empty mask", file=sys.stderr)
            else:
                if debug_reference:
                    print(f"    [DEBUG] idx {idx} found in reference_model", file=sys.stderr)
                reference_residual = reference_model[idx]
                if debug_reference:
                    print(f"    [DEBUG] reference_residual shape={reference_residual.shape}, dtype={reference_residual.dtype}, min={reference_residual.min()}, max={reference_residual.max()}", file=sys.stderr)
                
                # Compute difference: |residual_test - reference_residual|
                if debug_reference:
                    print(f"    [DEBUG] Computing difference (abs(residual - reference_residual))...", file=sys.stderr)
                diff = np.abs(residual.astype(np.int16) - reference_residual.astype(np.int16))
                diff = np.clip(diff, 0, 255).astype(np.uint8)
                if debug_reference:
                    print(f"    [DEBUG] diff shape={diff.shape}, dtype={diff.dtype}, min={diff.min()}, max={diff.max()}, mean={diff.mean():.2f}", file=sys.stderr)
                    print(f"    [DEBUG] threshold={reference_diff_threshold:.2f}", file=sys.stderr)
                
                # Threshold difference
                if debug_reference:
                    print(f"    [DEBUG] Thresholding difference...", file=sys.stderr)
                mask = (diff >= reference_diff_threshold).astype(np.uint8) * 255
                if debug_reference:
                    print(f"    [DEBUG] mask shape={mask.shape}, dtype={mask.dtype}, min={mask.min()}, max={mask.max()}, non-zero pixels={np.count_nonzero(mask)}", file=sys.stderr)
                
                # Morphology cleanup
                if debug_reference:
                    print(f"    [DEBUG] Applying morphology cleanup...", file=sys.stderr)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (resid_morph_ksize, resid_morph_ksize))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                if debug_reference:
                    print(f"    [DEBUG] After morphology: non-zero pixels={np.count_nonzero(mask)}", file=sys.stderr)
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
        
        # Debug artifacts for reference-residual detector
        # When debug_reference is enabled, save ALL images (ignore limit)
        print(f"\n[DEBUG] Crop {idx} processing for {image_name}", file=sys.stderr)
        print(f"  [DEBUG] Checking should_save_debug_ref conditions:", file=sys.stderr)
        print(f"    [DEBUG] debug_reference={debug_reference}", file=sys.stderr)
        print(f"    [DEBUG] output_dir is not None={output_dir is not None}", file=sys.stderr)
        print(f"    [DEBUG] image_name is not None={image_name is not None}", file=sys.stderr)
        print(f"    [DEBUG] detector == 'reference-residual'={detector == 'reference-residual'}", file=sys.stderr)
        
        should_save_debug_ref = (
            debug_reference 
            and output_dir is not None 
            and image_name is not None 
            and detector == "reference-residual"
        )
        print(f"  [DEBUG] should_save_debug_ref={should_save_debug_ref}", file=sys.stderr)
        
        if should_save_debug_ref:
            print(f"  [DEBUG] Checking array conditions:", file=sys.stderr)
            print(f"    [DEBUG] diff is not None={diff is not None}", file=sys.stderr)
            print(f"    [DEBUG] reference_residual is not None={reference_residual is not None}", file=sys.stderr)
            print(f"    [DEBUG] residual is not None={residual is not None}", file=sys.stderr)
            if reference_model:
                print(f"    [DEBUG] idx in reference_model={idx in reference_model}", file=sys.stderr)
                print(f"    [DEBUG] reference_model keys={list(reference_model.keys())}", file=sys.stderr)
            else:
                print(f"    [DEBUG] reference_model is None!", file=sys.stderr)
            
            if diff is not None and reference_residual is not None and residual is not None:
                print(f"  [DEBUG] All conditions met! Calling save_reference_debug_artifacts...", file=sys.stderr)
                try:
                    save_reference_debug_artifacts(
                        output_dir, f"{image_name}_crop{idx}",
                        crop, background, residual, reference_residual, diff, mask, blobs, reference_diff_threshold
                    )
                    debug_reference_count += 1
                    print(f"  [DEBUG] save_reference_debug_artifacts completed, count={debug_reference_count}", file=sys.stderr)
                    if debug_reference_count == 1:
                        debug_dir = output_dir / "debug_reference"
                        print(f"  [DEBUG] First debug image saved to {debug_dir}", file=sys.stderr)
                except Exception as e:
                    print(f"  [DEBUG] EXCEPTION in save_reference_debug_artifacts: {type(e).__name__}: {e}", file=sys.stderr)
                    import traceback
                    print(f"  [DEBUG] Traceback:", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
            else:
                # Diagnostic output for why images aren't being saved
                print(f"  [DEBUG] WARNING: Conditions NOT met for saving debug images!", file=sys.stderr)
                print(f"    [DEBUG] diff={diff is not None}, ref_residual={reference_residual is not None}, residual={residual is not None}", file=sys.stderr)
                if reference_model and idx not in reference_model:
                    print(f"    [DEBUG] Crop {idx} not found in reference_model!", file=sys.stderr)
                    print(f"    [DEBUG] Available crops in reference_model: {list(reference_model.keys())}", file=sys.stderr)
        else:
            print(f"  [DEBUG] should_save_debug_ref is False, skipping save", file=sys.stderr)
        
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
    
    # Print summary if debug is enabled
    if debug_reference and detector == "reference-residual" and output_dir is not None:
        if debug_reference_count > 0:
            print(f"  Saved {debug_reference_count} debug image set(s) for {image_name}", file=sys.stderr)
        else:
            print(f"  WARNING: No debug images were saved for {image_name}", file=sys.stderr)
    
    return results

