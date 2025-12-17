"""Core processing pipeline functions."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from config import DEFAULT_CROPS
from detection import detect_bright_spots
from image_io import load_image_optimized


def calculate_pixel_to_mm_ratio(
    crops: Sequence[Tuple[int, int, int, int]],
    module_width_mm: float,
    module_height_mm: float,
) -> Tuple[float, float]:
    """Calculate pixels per mm ratio for X and Y axes using crop rectangles as module boundaries.
    
    Args:
        crops: Sequence of crop rectangles (x, y, w, h) that define module boundaries
        module_width_mm: Module width in millimeters
        module_height_mm: Module height in millimeters
        
    Returns:
        Tuple of (pixels_per_mm_x, pixels_per_mm_y)
    """
    if not crops or len(crops) == 0:
        # If no crops, assume 1:1 ratio (fallback)
        return 1.0, 1.0
    
    # Use the first crop rectangle to calculate ratios
    # In practice, crops should define the module area (excluding black borders)
    x, y, w, h = crops[0]
    
    # Calculate pixels per mm for both axes
    pixels_per_mm_x = w / module_width_mm if module_width_mm > 0 else 1.0
    pixels_per_mm_y = h / module_height_mm if module_height_mm > 0 else 1.0
    
    return pixels_per_mm_x, pixels_per_mm_y


def get_crops_for_detection(
    img: Image.Image,
    crops: Optional[Sequence[Tuple[int, int, int, int]]],
    use_crops: bool,
) -> List[Tuple[int, int, int, int]]:
    """Get crops to use for detection. If use_crops is False, return whole image as single crop."""
    if not use_crops:
        # Return whole image as a single crop
        width, height = img.size
        return [(0, 0, width, height)]
    
    # Use provided crops or default crops
    if crops is None or len(crops) == 0:
        return DEFAULT_CROPS
    
    return list(crops)


def process_image(
    path: Path,
    crops: Optional[Sequence[Tuple[int, int, int, int]]],
    threshold: int,
    max_points: Optional[int] = None,
    use_crops: bool = True,
    load_pil: bool = True,
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
    good_images_dir: Optional[Path] = None,
    reference_diff_percentile: float = 99.5,
    reference_diff_min: int = 5,
    debug_reference: bool = False,
    debug_reference_max: int = 50,
    reference_model: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[Optional[Image.Image], List]:
    """Process a single image for bright spot detection.
    
    Uses optimized loading (OpenCV if available) for detection.
    PIL image is only loaded if load_pil=True (needed for annotation).
    
    Args:
        path: Path to image file
        crops: Crop rectangles to check
        threshold: Brightness threshold
        max_points: Max coordinates to extract
        use_crops: Whether to use crop regions
        load_pil: Whether to load PIL image (set False to save memory if not annotating)
        module_width_mm: Module width in millimeters (for physical measurements)
        module_height_mm: Module height in millimeters (for physical measurements)
        bright_spot_area_threshold: Minimum area threshold for bright spots (in mm² or cm²)
        area_unit: Unit for area threshold ("mm" for mm² or "cm" for cm²)
        dbscan_eps_mm: DBSCAN distance threshold in mm (default: 2.0mm)
        dbscan_min_samples: DBSCAN minimum samples per cluster (default: 3)
        good_images_dir: Directory containing GOOD reference images (for reference-residual detector)
        reference_diff_percentile: Percentile for thresholding difference image (0-100)
        reference_diff_min: Minimum difference threshold floor
        debug_reference: Enable debug output for reference-residual method
        debug_reference_max: Maximum number of images to generate debug artifacts for
    
    Returns:
        Tuple of (pil_image, results)
        - pil_image: PIL Image or None if load_pil=False
        - results: List of CropResult objects
    """
    from models import CropResult
    
    # Load image optimized for detection (OpenCV if available, much faster)
    gray_np, _ = load_image_optimized(path, as_grayscale=True)
    
    # Get image dimensions from numpy array for crop calculation
    height, width = gray_np.shape
    
    # Determine effective crops
    if not use_crops:
        effective_crops = [(0, 0, width, height)]
    elif crops is None or len(crops) == 0:
        effective_crops = DEFAULT_CROPS
    else:
        effective_crops = list(crops)
    
    # Perform detection on numpy array (fast, memory-efficient)
    image_name = path.stem if path else None
    
    # Debug output to trace debug_reference flag
    if detector == "reference-residual":
        print(f"[DEBUG] process_image: detector={detector}, debug_reference={debug_reference}, output_dir={output_dir}, image_name={image_name}, reference_model is None={reference_model is None}", file=sys.stderr)
    
    results = detect_bright_spots(
        gray_np,
        effective_crops,
        threshold,
        max_points=max_points,
        module_width_mm=module_width_mm,
        module_height_mm=module_height_mm,
        bright_spot_area_threshold=bright_spot_area_threshold,
        area_unit=area_unit,
        dbscan_eps_mm=dbscan_eps_mm,
        dbscan_min_samples=dbscan_min_samples,
        detector=detector,
        resid_blur_ksize=resid_blur_ksize,
        resid_percentile=resid_percentile,
        resid_min=resid_min,
        resid_morph_ksize=resid_morph_ksize,
        min_blob_area_mm2=min_blob_area_mm2,
        debug_residual=debug_residual,
        debug_max=debug_max,
        output_dir=output_dir,
        image_name=image_name,
        calculate_pixel_to_mm_ratio_func=calculate_pixel_to_mm_ratio,
        good_images_dir=good_images_dir,
        reference_diff_percentile=reference_diff_percentile,
        reference_diff_min=reference_diff_min,
        debug_reference=debug_reference,
        debug_reference_max=debug_reference_max,
        reference_model=reference_model,
    )
    
    # Load PIL image only if requested (for annotation)
    pil_img = None
    if load_pil:
        pil_img = Image.open(path)
    
    return pil_img, results

