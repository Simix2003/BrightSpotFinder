"""Image I/O utilities for bright spot detection."""

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image

from config import SUPPORTED_EXTS

# Try importing OpenCV for faster image loading (optional)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

# Global flag to enable/disable OpenCV optimization
USE_OPENCV = OPENCV_AVAILABLE  # Can be toggled for testing


def iter_images(images_dir: Path, recursive: bool = False) -> Iterable[Path]:
    """Optimized image discovery using os.scandir() for better performance on large directories.
    
    Args:
        images_dir: Directory to search for images
        recursive: If True, search subdirectories recursively (uses rglob)
    
    Yields:
        Path objects for supported image files
    """
    if recursive:
        # For recursive search, use rglob with pattern matching
        for ext in SUPPORTED_EXTS:
            for path in images_dir.rglob(f"*{ext}"):
                if path.is_file():
                    yield path
            for path in images_dir.rglob(f"*{ext.upper()}"):
                if path.is_file():
                    yield path
    else:
        # Use os.scandir() for much faster directory scanning
        try:
            with os.scandir(images_dir) as entries:
                # Collect paths first for sorting
                paths = []
                for entry in entries:
                    if entry.is_file():
                        path = Path(entry.path)
                        if path.suffix.lower() in SUPPORTED_EXTS:
                            paths.append(path)
                # Sort for consistent ordering
                for path in sorted(paths):
                    yield path
        except (OSError, PermissionError):
            # Fallback to glob if scandir fails
            for path in sorted(images_dir.glob("*")):
                if path.suffix.lower() in SUPPORTED_EXTS and path.is_file():
                    yield path


def ensure_output(output_dir: Path) -> None:
    """Ensure output directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)


def load_image_optimized(path: Path, as_grayscale: bool = True) -> Tuple[np.ndarray, Optional[Image.Image]]:
    """Load image optimized for processing.
    
    Uses OpenCV if available for faster loading, otherwise falls back to PIL.
    Returns numpy array and optionally the PIL Image object (for annotation).
    
    Args:
        path: Path to image file
        as_grayscale: If True, load as grayscale (faster for detection)
    
    Returns:
        Tuple of (numpy_array, pil_image)
        - numpy_array: Grayscale image as uint8 numpy array
        - pil_image: PIL Image object (None if OpenCV used and not needed)
    """
    if USE_OPENCV and as_grayscale:
        # Use OpenCV for fast grayscale loading
        gray_np = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray_np is None:
            raise ValueError(f"Failed to load image: {path}")
        return gray_np, None
    else:
        # Use PIL (original method, needed for annotation)
        img = Image.open(path)
        if as_grayscale:
            gray = img.convert("L")
            gray_np = np.array(gray)
            return gray_np, img
        else:
            return np.array(img), img

