import argparse
import csv
import json
import os
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Manager
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Try importing OpenCV for faster image loading (optional)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

# Global flag to enable/disable OpenCV optimization
USE_OPENCV = OPENCV_AVAILABLE  # Can be toggled for testing

# Optional web stack imports (only needed when running the API).
try:  # pragma: no cover - import guard for optional dependency
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - allow CLI to run without FastAPI installed
    BackgroundTasks = FastAPI = HTTPException = CORSMiddleware = BaseModel = Field = FileResponse = StaticFiles = StreamingResponse = None


# Rectangles are defined as (x, y, width, height).
DEFAULT_CROPS: List[Tuple[int, int, int, int]] = [
    # Example crops; adjust to your real regions of interest.
    (50, 50, 200, 200),
    (300, 200, 180, 180),
]

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Progress store for long-running batch jobs started via the API.
PROGRESS: Dict[str, Dict[str, object]] = {}

# Event streams for SSE (Server-Sent Events)
# Each job_id maps to a list of queue.Queue objects for SSE streams
STREAMS: Dict[str, List] = {}


class ProgressRenderer:
    """Minimal in-terminal progress bar for CLI runs."""

    def __init__(self, enable: bool = True, width: int = 40):
        self.enable = enable
        self.width = width
        self.reset(0)

    def reset(self, total: int) -> None:
        self.total = max(total, 0)
        self.good = 0
        self.no_good = 0
        self.with_bright = 0
        self.start = time.time()
        self.last_line = ""

    def update(self, current: int, *, has_bright: bool = False, is_no_good: bool = False) -> None:
        if not self.enable or self.total <= 0:
            return
        if has_bright:
            self.with_bright += 1
        if is_no_good:
            self.no_good += 1
        else:
            self.good += 1

        pct = current / self.total if self.total else 0
        filled = int(self.width * pct)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        rate = current / elapsed if elapsed > 0 else 0
        remaining = self.total - current
        eta = remaining / rate if rate > 0 else None

        line = (
            f"[{bar}] {current}/{self.total} "
            f"good:{self.good} no_good:{self.no_good} bright:{self.with_bright} "
            f"elapsed:{elapsed:.1f}s"
        )
        if eta is not None:
            line += f" ETA:{eta:.1f}s"

        # Minimize flicker by only rewriting when content changes
        if line != self.last_line:
            print("\r" + line, end="", flush=True)
            self.last_line = line

        if current >= self.total:
            print()



@dataclass
class Cluster:
    """Represents a cluster of bright points."""
    cluster_id: int
    points: List[Tuple[int, int]]
    area_mm2: float  # Area in mm²
    area_cm2: float  # Area in cm² (for convenience)
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)


@dataclass
class CropResult:
    crop_id: int
    rect: Tuple[int, int, int, int]
    bright_pixels: int
    total_pixels: int
    ratio: float
    bright_coords: Sequence[Tuple[int, int]]
    clusters: Optional[List[Cluster]] = None  # All detected clusters
    filtered_clusters: Optional[List[Cluster]] = None  # Clusters that pass area threshold
    cluster_count: int = 0
    filtered_cluster_count: int = 0
    total_bright_spot_area_mm2: float = 0.0  # Sum of all filtered cluster areas
    max_cluster_area_mm2: float = 0.0  # Area of the largest single cluster (for threshold checking)
    exceeds_threshold: bool = False  # True if any single cluster exceeds the area threshold


def load_crops(crops_file: Path | None) -> List[Tuple[int, int, int, int]]:
    """Load crops from JSON; fallback to defaults."""
    if crops_file is None:
        return DEFAULT_CROPS
    if not crops_file.exists():
        raise FileNotFoundError(f"Crops file not found: {crops_file}")
    with crops_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    crops: List[Tuple[int, int, int, int]] = []
    for item in data:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 4
            and all(isinstance(v, (int, float)) for v in item)
        ):
            x, y, w, h = item
            crops.append((int(x), int(y), int(w), int(h)))
        else:
            raise ValueError("Invalid crop entry in file; expected [x, y, w, h].")
    return crops


def save_crops(crops: Sequence[Sequence[int]], crops_file: Path) -> None:
    """Persist crops to disk for API/CLI use."""
    crops_file.parent.mkdir(parents=True, exist_ok=True)
    with crops_file.open("w", encoding="utf-8") as f:
        json.dump([list(map(int, rect)) for rect in crops], f, indent=2)


def load_module_config(module_config_file: Optional[Path] = None) -> Dict[str, float]:
    """Load module dimensions from JSON config file.
    
    Args:
        module_config_file: Optional path to module config JSON file
        
    Returns:
        Dictionary with module_width_mm and module_height_mm (defaults: 2100mm x 1050mm)
    """
    default_config = {
        "module_width_mm": 2100.0,
        "module_height_mm": 1050.0,
        "default_bright_spot_area_threshold_mm2": 100.0,
        "default_area_unit": "mm"
    }
    
    if module_config_file is None:
        return default_config
    
    if not module_config_file.exists():
        return default_config
    
    try:
        with module_config_file.open("r", encoding="utf-8") as f:
            config = json.load(f)
        # Merge with defaults, ensuring required keys exist
        result = default_config.copy()
        result.update(config)
        # Ensure numeric values
        result["module_width_mm"] = float(result.get("module_width_mm", 2100.0))
        result["module_height_mm"] = float(result.get("module_height_mm", 1050.0))
        return result
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # On error, return defaults
        return default_config


def tune_residual_params_interactive(
    image_path: Path,
    crop: Optional[Tuple[int, int, int, int]] = None,
    crops_file: Optional[Path] = None,
) -> None:
    """Interactive tool to tune residual detection parameters with real-time preview.
    
    Args:
        image_path: Path to the image to analyze
        crop: Optional crop rectangle (x, y, w, h). If None, uses first crop from crops_file or defaults.
        crops_file: Optional JSON file with crop definitions
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for interactive parameter tuning. "
            "Install with `pip install matplotlib`."
        ) from exc
    
    if not USE_OPENCV:
        raise SystemExit(
            "Residual detector requires OpenCV. Install with: pip install opencv-python"
        )
    
    # Load image
    gray_np, _ = load_image_optimized(image_path, as_grayscale=True)
    height, width = gray_np.shape
    
    # Determine crop to use
    if crop is not None:
        effective_crop = crop
    else:
        crops = load_crops(crops_file)
        if crops:
            effective_crop = crops[0]
        else:
            # Use whole image
            effective_crop = (0, 0, width, height)
    
    x, y, w, h = effective_crop
    x2, y2 = x + w, y + h
    
    # Validate crop bounds
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    x2 = max(x + 1, min(x2, width))
    y2 = max(y + 1, min(y2, height))
    w = x2 - x
    h = y2 - y
    
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid crop dimensions: {effective_crop} (image size: {width}x{height})")
    
    crop_img = gray_np[y:y2, x:x2]
    
    if crop_img.size == 0:
        raise ValueError(f"Crop resulted in empty image. Crop: {effective_crop}, Image size: {width}x{height}")
    
    print(f"Loaded crop: {x}, {y}, {w}, {h} (size: {crop_img.shape})", flush=True)
    print(f"Crop image range: [{crop_img.min()}, {crop_img.max()}]", flush=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Residual Parameter Tuning - {image_path.name}", fontsize=14, fontweight='bold')
    
    # Create subplots for different views
    ax_original = plt.subplot(2, 3, 1)
    ax_residual = plt.subplot(2, 3, 2)
    ax_mask = plt.subplot(2, 3, 3)
    ax_overlay = plt.subplot(2, 3, 4)
    ax_combined = plt.subplot(2, 3, (5, 6))
    
    # Display original crop
    im_original = ax_original.imshow(crop_img, cmap='gray')
    ax_original.set_title('Original Crop')
    ax_original.axis('off')
    
    # Placeholders for other images - set vmin/vmax explicitly for proper display
    im_residual = ax_residual.imshow(np.zeros_like(crop_img), cmap='gray', vmin=0, vmax=255)
    ax_residual.set_title('Residual Image')
    ax_residual.axis('off')
    
    im_mask = ax_mask.imshow(np.zeros_like(crop_img), cmap='gray', vmin=0, vmax=255)
    ax_mask.set_title('Binary Mask')
    ax_mask.axis('off')
    
    # For overlay, convert to RGB for color drawing
    overlay_base = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
    im_overlay = ax_overlay.imshow(overlay_base)
    ax_overlay.set_title('Overlay with Detections')
    ax_overlay.axis('off')
    
    # Combined view (original + residual side by side)
    combined_img = np.zeros((crop_img.shape[0], crop_img.shape[1] * 2), dtype=np.uint8)
    im_combined = ax_combined.imshow(combined_img, cmap='gray', vmin=0, vmax=255)
    ax_combined.set_title('Original (left) | Residual (right)')
    ax_combined.axis('off')
    
    # Adjust layout to make room for sliders
    plt.subplots_adjust(bottom=0.25, left=0.1, right=0.95, top=0.92, hspace=0.3, wspace=0.2)
    
    # Create sliders
    ax_blur = plt.axes([0.1, 0.18, 0.35, 0.02])
    ax_percentile = plt.axes([0.1, 0.15, 0.35, 0.02])
    ax_min = plt.axes([0.1, 0.12, 0.35, 0.02])
    ax_morph = plt.axes([0.1, 0.09, 0.35, 0.02])
    
    slider_blur = Slider(
        ax_blur, 'Blur KSize', 3, 201, valinit=51, valstep=2,
        valfmt='%d (must be odd)'
    )
    slider_percentile = Slider(
        ax_percentile, 'Percentile', 90.0, 100.0, valinit=99.7, valfmt='%.2f'
    )
    slider_min = Slider(
        ax_min, 'Min Residual', 0, 50, valinit=10, valfmt='%d'
    )
    slider_morph = Slider(
        ax_morph, 'Morph KSize', 1, 21, valinit=3, valstep=2,
        valfmt='%d (must be odd)'
    )
    
    # Info text
    info_text = fig.text(0.55, 0.15, '', fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def update_display(val=None):
        """Update all displays when sliders change."""
        try:
            # Get slider values
            blur_ksize = int(slider_blur.val)
            percentile = slider_percentile.val
            min_residual = int(slider_min.val)
            morph_ksize = int(slider_morph.val)
            
            # Ensure odd values
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            if morph_ksize % 2 == 0:
                morph_ksize += 1
            
            # Ensure minimum values
            blur_ksize = max(3, blur_ksize)
            morph_ksize = max(1, morph_ksize)
            
            # Compute residual
            background = cv2.GaussianBlur(crop_img, (blur_ksize, blur_ksize), 0)
            residual = cv2.subtract(crop_img, background)
            residual = np.clip(residual, 0, 255).astype(np.uint8)
            
            # Calculate threshold
            threshold = np.percentile(residual, percentile)
            threshold = max(threshold, min_residual)
            
            # Create mask
            mask = (residual >= threshold).astype(np.uint8) * 255
            
            # Morphology cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
            mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
            
            # Normalize residual for display
            residual_min = residual.min()
            residual_max = residual.max()
            if residual_max > residual_min:
                residual_norm = ((residual - residual_min) / 
                                (residual_max - residual_min + 1e-6) * 255).astype(np.uint8)
            else:
                # If residual is constant, show it as-is (might be all zeros)
                residual_norm = residual.astype(np.uint8)
            
            # Debug: print residual stats
            if residual_norm.max() == 0:
                print(f"WARNING: Residual normalized image is all zeros. Raw residual range: [{residual_min}, {residual_max}]", flush=True)
            
            # Update residual image - ensure it's visible
            im_residual.set_data(residual_norm)
            im_residual.set_clim(vmin=0, vmax=255)  # Explicitly set color limits
            ax_residual.set_title(f'Residual (min={residual.min()}, max={residual.max()}, threshold={threshold:.1f})')
            
            # Update mask - ensure it's visible
            im_mask.set_data(mask_cleaned)
            im_mask.set_clim(vmin=0, vmax=255)  # Explicitly set color limits
            bright_pixels = np.count_nonzero(mask_cleaned)
            total_pixels = mask_cleaned.size
            ratio = bright_pixels / total_pixels if total_pixels > 0 else 0
            ax_mask.set_title(f'Mask ({bright_pixels} pixels, {ratio*100:.2f}%)')
            
            # Create overlay with detected blobs - use RGB for color
            overlay = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
            # Find contours for visualization
            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 10:  # Filter small noise
                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)
                    cv2.rectangle(overlay, (x_rect, y_rect), 
                                 (x_rect + w_rect, y_rect + h_rect), (0, 255, 0), 2)
                    # Draw centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
            
            im_overlay.set_data(overlay)
            ax_overlay.set_title(f'Overlay ({len(contours)} blobs)')
            
            # Update combined view
            combined_img[:, :crop_img.shape[1]] = crop_img
            combined_img[:, crop_img.shape[1]:] = residual_norm
            im_combined.set_data(combined_img)
            im_combined.set_clim(vmin=0, vmax=255)  # Explicitly set color limits
            
            # Update info text
            info = (
                f"Parameters:\n"
                f"  Blur KSize: {blur_ksize}\n"
                f"  Percentile: {percentile:.2f}\n"
                f"  Min Residual: {min_residual}\n"
                f"  Morph KSize: {morph_ksize}\n"
                f"  Threshold: {threshold:.2f}\n"
                f"  Residual Range: [{residual.min()}, {residual.max()}]\n"
                f"  Bright Pixels: {bright_pixels} ({ratio*100:.2f}%)\n"
                f"  Detected Blobs: {len(contours)}"
            )
            info_text.set_text(info)
            
            fig.canvas.draw_idle()
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            info_text.set_text(error_msg)
            print(error_msg, flush=True)
            fig.canvas.draw_idle()
    
    # Connect sliders to update function
    slider_blur.on_changed(update_display)
    slider_percentile.on_changed(update_display)
    slider_min.on_changed(update_display)
    slider_morph.on_changed(update_display)
    
    # Initial update
    update_display()
    
    # Add button to save current parameters
    from matplotlib.widgets import Button
    ax_save = plt.axes([0.55, 0.09, 0.15, 0.03])
    button_save = Button(ax_save, 'Save Parameters')
    
    def save_params(event):
        blur_ksize = int(slider_blur.val)
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        blur_ksize = max(3, blur_ksize)
        
        morph_ksize = int(slider_morph.val)
        if morph_ksize % 2 == 0:
            morph_ksize += 1
        morph_ksize = max(1, morph_ksize)
        
        params = {
            "resid_blur_ksize": blur_ksize,
            "resid_percentile": slider_percentile.val,
            "resid_min": int(slider_min.val),
            "resid_morph_ksize": morph_ksize,
        }
        
        import json
        params_file = image_path.parent / f"{image_path.stem}_residual_params.json"
        with params_file.open("w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        
        info_text.set_text(f"Parameters saved to:\n{params_file}\n\n{info_text.get_text()}")
        fig.canvas.draw_idle()
        print(f"\nParameters saved to: {params_file}")
        print(json.dumps(params, indent=2))
    
    button_save.on_clicked(save_params)
    
    # Add button to copy command
    ax_copy = plt.axes([0.72, 0.09, 0.15, 0.03])
    button_copy = Button(ax_copy, 'Copy Command')
    
    def copy_command(event):
        blur_ksize = int(slider_blur.val)
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        blur_ksize = max(3, blur_ksize)
        
        morph_ksize = int(slider_morph.val)
        if morph_ksize % 2 == 0:
            morph_ksize += 1
        morph_ksize = max(1, morph_ksize)
        
        cmd = (
            f"--resid-blur-ksize {blur_ksize} "
            f"--resid-percentile {slider_percentile.val:.2f} "
            f"--resid-min {int(slider_min.val)} "
            f"--resid-morph-ksize {morph_ksize}"
        )
        
        try:
            import pyperclip
            pyperclip.copy(cmd)
            info_text.set_text(f"Command copied to clipboard!\n\n{cmd}\n\n{info_text.get_text()}")
        except ImportError:
            info_text.set_text(f"Command (install pyperclip to auto-copy):\n\n{cmd}\n\n{info_text.get_text()}")
        
        fig.canvas.draw_idle()
        print(f"\nCommand line arguments:\n{cmd}")
    
    button_copy.on_clicked(copy_command)
    
    plt.show()


def define_crops_interactive(image_path: Path) -> List[Tuple[int, int, int, int]]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import RectangleSelector
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for interactive crop definition. "
            "Install with `pip install matplotlib`."
        ) from exc

    img = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(
        "Drag to draw rectangles. Close window or press Enter to finish.\n"
        "Left click-drag creates a rectangle; right click removes last."
    )

    crops: List[Tuple[int, int, int, int]] = []
    drawn: List[Rectangle] = []

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        rect = (int(x), int(y), int(w), int(h))
        crops.append(rect)
        patch = Rectangle((x, y), w, h, linewidth=2, edgecolor="yellow", facecolor="none")
        drawn.append(patch)
        ax.add_patch(patch)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)
        elif event.key == "backspace" and crops:
            # Remove last rectangle.
            crops.pop()
            patch = drawn.pop()
            patch.remove()
            fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],  # left click
        spancoords="pixels",
        interactive=True,
        minspanx=1,
        minspany=1,
    )
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    selector.set_active(False)
    return crops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect bright spots in predefined regions of images."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images"),
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write annotated images and CSV summary.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=228,
        help="Fixed grayscale threshold (0-255) to consider a pixel bright.",
    )
    parser.add_argument(
        "--max-dots",
        type=int,
        default=1000,
        help="Maximum number of bright pixels to draw to avoid huge outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotated images.",
    )
    parser.add_argument(
        "--crops-file",
        type=Path,
        help="Optional JSON file with a list of [x, y, w, h] crops to use instead of defaults.",
    )
    parser.add_argument(
        "--define-crops",
        type=Path,
        help="Open an interactive window to draw rectangles on the given image and output crop coordinates.",
    )
    parser.add_argument(
        "--save-crops-json",
        type=Path,
        help="When used with --define-crops, save drawn rectangles to this JSON file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Optional: process images in chunks of this size, writing each batch to a subfolder under output. 0 = no batching.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["threshold", "residual"],
        default="threshold",
        help="Detection method: 'threshold' (fixed threshold) or 'residual' (background subtraction).",
    )
    parser.add_argument(
        "--resid-blur-ksize",
        type=int,
        default=51,
        help="Gaussian blur kernel size for residual method (must be odd, >= 3).",
    )
    parser.add_argument(
        "--resid-percentile",
        type=float,
        default=99.7,
        help="Percentile for adaptive threshold in residual method (0-100).",
    )
    parser.add_argument(
        "--resid-min",
        type=int,
        default=10,
        help="Minimum residual value floor for residual method.",
    )
    parser.add_argument(
        "--resid-morph-ksize",
        type=int,
        default=3,
        help="Morphology kernel size for cleanup in residual method (must be odd, >= 1).",
    )
    parser.add_argument(
        "--min-blob-area-mm2",
        type=float,
        default=None,
        help="Minimum blob area in mm² (alternative/additional to bright_spot_area_threshold).",
    )
    parser.add_argument(
        "--debug-residual",
        action="store_true",
        help="Enable debug output for residual method (saves debug images).",
    )
    parser.add_argument(
        "--debug-max",
        type=int,
        default=50,
        help="Maximum number of images to generate debug artifacts for (when --debug-residual is used).",
    )
    parser.add_argument(
        "--residual-only",
        action="store_true",
        help="Only generate residual debug images (skip annotated images and CSV). Implies --detector=residual and --debug-residual.",
    )
    parser.add_argument(
        "--tune-residual",
        type=Path,
        help="Open interactive parameter tuning GUI for residual detection on the specified image.",
    )
    parser.add_argument(
        "--tune-crop-index",
        type=int,
        default=0,
        help="Crop index to use for tuning (when --tune-residual is used, default: 0).",
    )
    
    args = parser.parse_args()
    
    # If residual-only mode, force detector to residual and enable debug
    if args.residual_only:
        args.detector = "residual"
        args.debug_residual = True
        args.debug_max = 999999  # Process all images
    
    # Validation
    if args.resid_blur_ksize < 3:
        parser.error("--resid-blur-ksize must be >= 3")
    if args.resid_blur_ksize % 2 == 0:
        args.resid_blur_ksize += 1  # Make odd
        print(f"Warning: --resid-blur-ksize adjusted to {args.resid_blur_ksize} (must be odd)", file=sys.stderr)
    
    if not (0 <= args.resid_percentile <= 100):
        parser.error("--resid-percentile must be in range [0, 100]")
    
    if args.resid_morph_ksize < 1:
        parser.error("--resid-morph-ksize must be >= 1")
    if args.resid_morph_ksize % 2 == 0:
        args.resid_morph_ksize += 1  # Make odd
        print(f"Warning: --resid-morph-ksize adjusted to {args.resid_morph_ksize} (must be odd)", file=sys.stderr)
    
    return args


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
    output_dir.mkdir(parents=True, exist_ok=True)


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
        pixels_per_mm_x, pixels_per_mm_y = calculate_pixel_to_mm_ratio(
            crops, module_width_mm, module_height_mm
        )
    
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


def annotate_image(
    img: Image.Image,
    results: Sequence[CropResult],
    max_dots: int,
    show_crops: bool = True,
) -> Image.Image:
    annotated = img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    for res in results:
        x, y, w, h = res.rect
        if show_crops:
            # Outline the crop area for reference.
            draw.rectangle([x, y, x + w, y + h], outline="yellow", width=2)

        if res.bright_pixels == 0:
            continue

        coords = res.bright_coords
        if len(coords) > max_dots:
            # Downsample evenly to avoid excessive drawing.
            step = max(1, len(coords) // max_dots)
            coords = coords[::step]

        for cx, cy in coords:
            # Small red dot at the bright pixel location.
            draw.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill="red")

    return annotated


def write_csv(
    output_csv: Path,
    per_image: Iterable[Dict[str, object]],
    bright_spot_area_threshold: Optional[float] = None,
    area_unit: str = "mm",
    detector: str = "threshold",
    residual_threshold: Optional[float] = None,
) -> None:
    """Write CSV with overall stats and per-module rows for NO GOOD modules only."""
    per_image_list = list(per_image)

    total_modules = len(per_image_list)
    good_modules = 0  # modules with zero bright pixels and no threshold breach
    no_good_modules = 0  # modules with any bright pixels or threshold breach

    no_good_rows: List[List[object]] = []

    total_area_mm2 = 0.0  # aggregate area over NO GOOD modules
    modules_with_area = 0  # NO GOOD modules with >0 area

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Section 1: per-module stats
        writer.writerow(
            [
                "ID_MODULO",
                "Filename",
                "Detector",
                "Total_Bright_Pixels",
                "Bright_Spot_Area_mm2",
                "Bright_Spot_Area_cm2",
                "Cluster_Count",
                "Filtered_Cluster_Count",
                "Max_Cluster_Area_mm2",
                "Residual_Threshold",
                "Status",
                "Exceeds_Threshold",
            ]
        )

        for item in per_image_list:
            filename = item.get("filename", "")
            id_modulo = Path(filename).stem[:16]
            bright_total = int(item.get("bright_total", 0) or 0)
            crops = item.get("crops", [])

            # Aggregate per-image stats from crops
            bright_spot_area_mm2 = sum(
                float(c.get("total_bright_spot_area_mm2", 0.0) or 0.0) for c in crops
            )
            bright_spot_area_cm2 = bright_spot_area_mm2 / 100.0
            cluster_count = sum(int(c.get("cluster_count", 0) or 0) for c in crops)
            filtered_cluster_count = sum(
                int(c.get("filtered_cluster_count", 0) or 0) for c in crops
            )
            max_cluster_area_mm2 = 0.0
            for c in crops:
                max_cluster_area_mm2 = max(
                    max_cluster_area_mm2, float(c.get("max_cluster_area_mm2", 0.0) or 0.0)
                )
            exceeds_threshold = any(bool(c.get("exceeds_threshold", False)) for c in crops)

            # Determine NO GOOD: either threshold breach or any bright pixels
            is_no_good = exceeds_threshold or bright_total > 0
            if is_no_good:
                no_good_modules += 1
                total_area_mm2 += bright_spot_area_mm2
                if bright_spot_area_mm2 > 0:
                    modules_with_area += 1

                # Get detector from item metadata (default to passed detector)
                item_detector = item.get("detector", detector)
                item_residual_threshold = item.get("residual_threshold", residual_threshold)
                
                no_good_rows.append(
                    [
                        id_modulo,
                        filename,
                        item_detector,
                        bright_total,
                        f"{bright_spot_area_mm2:.2f}",
                        f"{bright_spot_area_cm2:.2f}",
                        cluster_count,
                        filtered_cluster_count,
                        f"{max_cluster_area_mm2:.2f}",
                        f"{item_residual_threshold:.2f}" if item_residual_threshold is not None else "",
                        "No Good",
                        exceeds_threshold,
                    ]
                )
            else:
                good_modules += 1

        # Write only NO GOOD modules
        for row in no_good_rows:
            writer.writerow(row)

        # Section 2: summary totals
        writer.writerow([])
        writer.writerow(["Statistic", "Value"])
        writer.writerow(["Total_Modules", total_modules])
        writer.writerow(["Good_Modules", good_modules])
        writer.writerow(["No_Good_Modules", no_good_modules])
        writer.writerow(["Detector", detector])
        writer.writerow(["Bright_Spot_Area_Threshold", bright_spot_area_threshold or ""])
        writer.writerow(["Area_Unit", area_unit])
        if residual_threshold is not None:
            writer.writerow(["Residual_Threshold", f"{residual_threshold:.2f}"])
        writer.writerow(["Total_Bright_Spot_Area_mm2", f"{total_area_mm2:.2f}"])
        writer.writerow(["Total_Bright_Spot_Area_cm2", f"{(total_area_mm2/100.0):.2f}"])
        avg_area = total_area_mm2 / modules_with_area if modules_with_area > 0 else 0.0
        writer.writerow(["Average_Bright_Spot_Area_mm2", f"{avg_area:.2f}"])


def process_batch(
    images_dir: Path,
    output_dir: Path,
    crops: Optional[Sequence[Tuple[int, int, int, int]]],
    threshold: int,
    max_dots: int,
    overwrite: bool,
    progress_cb: Optional[Callable[[int, int, float, Optional[float]], None]] = None,
    max_workers: Optional[int] = None,
    use_crops: bool = True,
    module_width_mm: Optional[float] = None,
    module_height_mm: Optional[float] = None,
    bright_spot_area_threshold: Optional[float] = None,
    area_unit: str = "mm",
    dbscan_eps_mm: float = 2.0,
    dbscan_min_samples: int = 3,
    progress_renderer: Optional[ProgressRenderer] = None,
    image_paths_override: Optional[List[Path]] = None,
    detector: str = "threshold",
    resid_blur_ksize: int = 51,
    resid_percentile: float = 99.7,
    resid_min: int = 10,
    resid_morph_ksize: int = 3,
    min_blob_area_mm2: Optional[float] = None,
    debug_residual: bool = False,
    debug_max: int = 50,
    residual_only: bool = False,
) -> Dict[str, object]:
    """Batch process images; returns summary payload for CLI/API.
    
    Args:
        images_dir: Directory containing input images
        output_dir: Directory to write annotated images and CSV summary
        crops: Sequence of crop rectangles (x, y, w, h)
        threshold: Grayscale threshold for bright pixel detection
        max_dots: Maximum number of bright pixels to draw
        overwrite: Whether to overwrite existing annotated images
        progress_cb: Optional callback for progress updates
            Signature: (current: int, total: int, elapsed: float, estimated_remaining: Optional[float])
        max_workers: Number of parallel workers (None = CPU count, 1 = sequential)
        use_crops: Whether to use crop regions
        module_width_mm: Module width in millimeters (for physical measurements)
        module_height_mm: Module height in millimeters (for physical measurements)
        bright_spot_area_threshold: Minimum area threshold for bright spots (in mm² or cm²)
        area_unit: Unit for area threshold ("mm" for mm² or "cm" for cm²)
        dbscan_eps_mm: DBSCAN distance threshold in mm (default: 2.0mm)
        dbscan_min_samples: DBSCAN minimum samples per cluster (default: 3)
        residual_only: If True, only generate residual debug images (skip annotated images and CSV)
    
    Returns:
        Dictionary with processing summary
    """
    ensure_output(output_dir)
    # Only create output directories if not in residual-only mode
    if not residual_only:
        noted_dir = output_dir / "Noted"
        images_dir_out = output_dir / "Images"
        ensure_output(noted_dir)
        ensure_output(images_dir_out)

    image_paths = list(image_paths_override) if image_paths_override is not None else list(iter_images(images_dir))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Determine if we should use parallel processing
    use_parallel = max_workers is None or max_workers > 1
    if max_workers is None:
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
    elif max_workers < 1:
        max_workers = 1
        use_parallel = False

    if progress_renderer is not None:
        progress_renderer.reset(len(image_paths))

    if use_parallel and len(image_paths) > 1:
        return _process_batch_parallel(
            image_paths=image_paths,
            output_dir=output_dir,
            crops=crops,
            threshold=threshold,
            max_dots=max_dots,
            overwrite=overwrite,
            progress_cb=progress_cb,
            max_workers=max_workers,
            use_crops=use_crops,
            module_width_mm=module_width_mm,
            module_height_mm=module_height_mm,
            bright_spot_area_threshold=bright_spot_area_threshold,
            area_unit=area_unit,
            dbscan_eps_mm=dbscan_eps_mm,
            dbscan_min_samples=dbscan_min_samples,
            progress_renderer=progress_renderer,
            detector=detector,
            resid_blur_ksize=resid_blur_ksize,
            resid_percentile=resid_percentile,
            resid_min=resid_min,
            resid_morph_ksize=resid_morph_ksize,
            min_blob_area_mm2=min_blob_area_mm2,
            debug_residual=debug_residual,
            debug_max=debug_max,
            residual_only=residual_only,
        )
    else:
        # Sequential processing (original implementation)
        return _process_batch_sequential(
            image_paths=image_paths,
            output_dir=output_dir,
            crops=crops,
            threshold=threshold,
            max_dots=max_dots,
            overwrite=overwrite,
            progress_cb=progress_cb,
            use_crops=use_crops,
            module_width_mm=module_width_mm,
            module_height_mm=module_height_mm,
            bright_spot_area_threshold=bright_spot_area_threshold,
            area_unit=area_unit,
            dbscan_eps_mm=dbscan_eps_mm,
            dbscan_min_samples=dbscan_min_samples,
            progress_renderer=progress_renderer,
            detector=detector,
            resid_blur_ksize=resid_blur_ksize,
            resid_percentile=resid_percentile,
            resid_min=resid_min,
            resid_morph_ksize=resid_morph_ksize,
            min_blob_area_mm2=min_blob_area_mm2,
            debug_residual=debug_residual,
            debug_max=debug_max,
            residual_only=residual_only,
        )


def _process_batch_sequential(
    image_paths: List[Path],
    output_dir: Path,
    crops: Optional[Sequence[Tuple[int, int, int, int]]],
    threshold: int,
    max_dots: int,
    overwrite: bool,
    progress_cb: Optional[Callable[[int, int, float, Optional[float]], None]] = None,
    use_crops: bool = True,
    module_width_mm: Optional[float] = None,
    module_height_mm: Optional[float] = None,
    bright_spot_area_threshold: Optional[float] = None,
    area_unit: str = "mm",
    dbscan_eps_mm: float = 2.0,
    dbscan_min_samples: int = 3,
    progress_renderer: Optional[ProgressRenderer] = None,
    detector: str = "threshold",
    resid_blur_ksize: int = 51,
    resid_percentile: float = 99.7,
    resid_min: int = 10,
    resid_morph_ksize: int = 3,
    min_blob_area_mm2: Optional[float] = None,
    debug_residual: bool = False,
    debug_max: int = 50,
    residual_only: bool = False,
) -> Dict[str, object]:
    """Sequential batch processing (original implementation)."""
    if not residual_only:
        noted_dir = output_dir / "Noted"
        images_dir_out = output_dir / "Images"
    
    found_filenames: List[str] = []
    per_image: List[Dict[str, object]] = []
    
    start_time = time.time()

    for idx, path in enumerate(image_paths):
        img, results = process_image(
            path=path,
            crops=crops,
            threshold=threshold,
            max_points=max_dots,
            use_crops=use_crops,
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
        )
        bright_total = sum(res.bright_pixels for res in results)
        # Serialize crop results with cluster information
        crops_data = []
        for r in results:
            crop_data = {
                "crop_id": r.crop_id,
                "rect": r.rect,
                "bright_pixels": r.bright_pixels,
                "total_pixels": r.total_pixels,
                "ratio": r.ratio,
                "cluster_count": r.cluster_count,
                "filtered_cluster_count": r.filtered_cluster_count,
                "total_bright_spot_area_mm2": r.total_bright_spot_area_mm2,
                "max_cluster_area_mm2": r.max_cluster_area_mm2,
                "exceeds_threshold": r.exceeds_threshold,
            }
            # Add cluster information if available
            if r.filtered_clusters is not None and len(r.filtered_clusters) > 0:
                crop_data["filtered_clusters"] = [
                    {
                        "cluster_id": c.cluster_id,
                        "points": c.points,  # Include points for visualization
                        "area_mm2": c.area_mm2,
                        "area_cm2": c.area_cm2,
                        "centroid": c.centroid,
                        "bounding_box": c.bounding_box,
                    }
                    for c in r.filtered_clusters
                ]
            crops_data.append(crop_data)
        
        per_image.append(
            {
                "filename": path.name,
                "bright_total": bright_total,
                "crops": crops_data,
                "detector": detector,
            }
        )

        # Skip saving annotated images and copies in residual-only mode
        if not residual_only and bright_total > 0:
            found_filenames.append(path.name)

            noted_path = noted_dir / path.name
            if overwrite or not noted_path.exists():
                annotated = annotate_image(
                    img, results, max_dots=max_dots, show_crops=False
                )
                annotated.save(noted_path)

            raw_copy_path = images_dir_out / path.name
            if overwrite or not raw_copy_path.exists():
                shutil.copy2(path, raw_copy_path)
        elif residual_only:
            # In residual-only mode, we still track files for progress
            found_filenames.append(path.name)

        if progress_renderer is not None:
            has_bright = bright_total > 0
            is_no_good = any(c.get("exceeds_threshold") for c in crops_data)
            progress_renderer.update(idx + 1, has_bright=has_bright, is_no_good=is_no_good)

        if progress_cb is not None:
            current = idx + 1
            total = len(image_paths)
            elapsed = time.time() - start_time
            
            # Calculate estimated remaining time
            estimated_remaining = None
            if current > 0:
                avg_time_per_image = elapsed / current
                remaining_images = total - current
                estimated_remaining = avg_time_per_image * remaining_images
            
            progress_cb(current, total, elapsed, estimated_remaining)

    # Skip CSV writing in residual-only mode
    if not residual_only:
        write_csv(
            output_dir / "summary.csv",
            per_image,
            bright_spot_area_threshold=bright_spot_area_threshold,
            area_unit=area_unit,
            detector=detector,
        )
    return {
        "processed": len(image_paths),
        "with_bright": len(found_filenames),
        "found_filenames": found_filenames,
        "details": per_image,
        "output_dir": str(output_dir),
    }


def _process_batch_parallel(
    image_paths: List[Path],
    output_dir: Path,
    crops: Optional[Sequence[Tuple[int, int, int, int]]],
    threshold: int,
    max_dots: int,
    overwrite: bool,
    progress_cb: Optional[Callable[[int, int, float, Optional[float]], None]] = None,
    max_workers: int = 1,
    use_crops: bool = True,
    module_width_mm: Optional[float] = None,
    module_height_mm: Optional[float] = None,
    bright_spot_area_threshold: Optional[float] = None,
    area_unit: str = "mm",
    dbscan_eps_mm: float = 2.0,
    dbscan_min_samples: int = 3,
    progress_renderer: Optional[ProgressRenderer] = None,
    detector: str = "threshold",
    resid_blur_ksize: int = 51,
    resid_percentile: float = 99.7,
    resid_min: int = 10,
    resid_morph_ksize: int = 3,
    min_blob_area_mm2: Optional[float] = None,
    debug_residual: bool = False,
    debug_max: int = 50,
    residual_only: bool = False,
) -> Dict[str, object]:
    """Parallel batch processing using multiprocessing."""
    # Prepare arguments for workers
    worker_args = [
        (path, crops, threshold, max_dots, output_dir, overwrite, use_crops,
         module_width_mm, module_height_mm, bright_spot_area_threshold,
         area_unit, dbscan_eps_mm, dbscan_min_samples,
         detector, resid_blur_ksize, resid_percentile, resid_min, resid_morph_ksize,
         min_blob_area_mm2, debug_residual, debug_max, residual_only)
        for path in image_paths
    ]
    
    found_filenames: List[str] = []
    per_image: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []
    
    start_time = time.time()
    completed_count = 0
    total = len(image_paths)
    
    # Use Manager for thread-safe progress tracking
    manager = Manager()
    progress_queue = manager.Queue()
    stop_event = manager.Event()
    
    def progress_monitor():
        """Monitor progress queue and call progress callback."""
        nonlocal completed_count
        while not stop_event.is_set() and completed_count < total:
            try:
                progress_queue.get(timeout=0.5)
                completed_count += 1
                if progress_cb is not None:
                    elapsed = time.time() - start_time
                    estimated_remaining = None
                    if completed_count > 0:
                        avg_time_per_image = elapsed / completed_count
                        remaining_images = total - completed_count
                        estimated_remaining = avg_time_per_image * remaining_images
                    progress_cb(completed_count, total, elapsed, estimated_remaining)
            except:
                # Timeout - check if we're done
                if completed_count >= total:
                    break
                continue
    
    # Start progress monitor in a separate thread
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()
    
    # Process images in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_image_worker, args): args[0]
            for args in worker_args
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_path):
            result = None
            try:
                result = future.result()
                per_image.append(result)
                
                if result.get("error"):
                    errors.append({
                        "filename": result["filename"],
                        "error": result["error"],
                    })
                elif result.get("bright_total", 0) > 0:
                    found_filenames.append(result["filename"])
                
                # Signal progress
                progress_queue.put(1)
            except Exception as exc:
                path = future_to_path[future]
                error_info = {
                    "filename": path.name,
                    "error": str(exc),
                }
                errors.append(error_info)
                result = {
                    "filename": path.name,
                    "bright_total": 0,
                    "crops": [],
                    "error": str(exc),
                    "detector": detector,  # Get from function scope
                }
                per_image.append(result)
                progress_queue.put(1)

            if progress_renderer is not None:
                current_count = len(per_image)
                has_bright = result.get("bright_total", 0) > 0 if result is not None else False
                is_no_good = False
                crops = result.get("crops", []) if result is not None else []
                if crops:
                    is_no_good = any(c.get("exceeds_threshold") for c in crops)
                progress_renderer.update(current_count, has_bright=has_bright, is_no_good=is_no_good)
    
    # Signal completion and wait for progress monitor to finish
    stop_event.set()
    monitor_thread.join(timeout=2.0)
    
    # Sort results by original order (by filename to match original sequence)
    filename_to_index = {path.name: idx for idx, path in enumerate(image_paths)}
    per_image.sort(key=lambda x: filename_to_index.get(x["filename"], 0))
    
    # Skip CSV writing in residual-only mode
    if not residual_only:
        write_csv(
            output_dir / "summary.csv",
            per_image,
            bright_spot_area_threshold=bright_spot_area_threshold,
            area_unit=area_unit,
            detector=detector,
        )
    
    result = {
        "processed": len(image_paths),
        "with_bright": len(found_filenames),
        "found_filenames": found_filenames,
        "details": per_image,
        "output_dir": str(output_dir),
    }
    
    if errors:
        result["errors"] = errors
    
    return result


def detect_single(
    path: Path,
    crops: Optional[Sequence[Tuple[int, int, int, int]]],
    threshold: int,
    max_dots: int,
    output_dir: Optional[Path] = None,
    use_crops: bool = True,
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
) -> Dict[str, object]:
    """Process one image and optionally emit an annotated copy."""
    img, results = process_image(
        path=path,
        crops=crops,
        threshold=threshold,
        max_points=max_dots,
        use_crops=use_crops,
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
    )
    annotated_path: Optional[Path] = None

    if output_dir:
        ensure_output(output_dir)
        annotated = annotate_image(img, results, max_dots=max_dots, show_crops=False)
        annotated_path = output_dir / f"live_{path.name}"
        annotated.save(annotated_path)

    # Serialize results with cluster information
    crops_data = []
    for r in results:
        crop_data = {
            "crop_id": r.crop_id,
            "rect": r.rect,
            "bright_pixels": r.bright_pixels,
            "total_pixels": r.total_pixels,
            "ratio": r.ratio,
            "bright_coords": list(r.bright_coords),
            "cluster_count": r.cluster_count,
            "filtered_cluster_count": r.filtered_cluster_count,
            "total_bright_spot_area_mm2": r.total_bright_spot_area_mm2,
        }
        # Add cluster information if available
        if r.clusters is not None:
            crop_data["clusters"] = [
                {
                    "cluster_id": c.cluster_id,
                    "points": c.points,
                    "area_mm2": c.area_mm2,
                    "area_cm2": c.area_cm2,
                    "centroid": c.centroid,
                    "bounding_box": c.bounding_box,
                }
                for c in r.clusters
            ]
        if r.filtered_clusters is not None:
            crop_data["filtered_clusters"] = [
                {
                    "cluster_id": c.cluster_id,
                    "points": c.points,
                    "area_mm2": c.area_mm2,
                    "area_cm2": c.area_cm2,
                    "centroid": c.centroid,
                    "bounding_box": c.bounding_box,
                }
                for c in r.filtered_clusters
            ]
        crops_data.append(crop_data)
    
    return {
        "filename": path.name,
        "annotated_path": str(annotated_path) if annotated_path else None,
        "crops": crops_data,
    }


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
) -> Tuple[Optional[Image.Image], List[CropResult]]:
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
    
    Returns:
        Tuple of (pil_image, results)
        - pil_image: PIL Image or None if load_pil=False
        - results: List of CropResult objects
    """
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
    )
    
    # Load PIL image only if requested (for annotation)
    pil_img = None
    if load_pil:
        pil_img = Image.open(path)
    
    return pil_img, results


def process_image_worker(args: Tuple) -> Dict[str, object]:
    """Worker function for parallel processing.
    
    Args:
        args: Tuple of (path, crops, threshold, max_dots, output_dir, overwrite, use_crops,
                        module_width_mm, module_height_mm, bright_spot_area_threshold,
                        area_unit, dbscan_eps_mm, dbscan_min_samples,
                        detector, resid_blur_ksize, resid_percentile, resid_min, resid_morph_ksize,
                        min_blob_area_mm2, debug_residual, debug_max, residual_only)
    
    Returns:
        Dictionary with image processing results or error information
    """
    (path, crops, threshold, max_dots, output_dir, overwrite, use_crops,
     module_width_mm, module_height_mm, bright_spot_area_threshold,
     area_unit, dbscan_eps_mm, dbscan_min_samples,
     detector, resid_blur_ksize, resid_percentile, resid_min, resid_morph_ksize,
     min_blob_area_mm2, debug_residual, debug_max, residual_only) = args
    try:
        # Memory optimization: only load PIL image if we need to annotate
        # First, do detection without loading full PIL image
        _, results = process_image(
            path=path,
            crops=crops,
            threshold=threshold,
            max_points=max_dots,
            use_crops=use_crops,
            load_pil=False,
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
        )
        bright_total = sum(res.bright_pixels for res in results)
        
        # Serialize crop results with cluster information
        crops_data = []
        for r in results:
            crop_data = {
                "crop_id": r.crop_id,
                "rect": r.rect,
                "bright_pixels": r.bright_pixels,
                "total_pixels": r.total_pixels,
                "ratio": r.ratio,
                "cluster_count": r.cluster_count,
                "filtered_cluster_count": r.filtered_cluster_count,
                "total_bright_spot_area_mm2": r.total_bright_spot_area_mm2,
                "max_cluster_area_mm2": r.max_cluster_area_mm2,
                "exceeds_threshold": r.exceeds_threshold,
            }
            # Add cluster information if available
            if r.filtered_clusters is not None and len(r.filtered_clusters) > 0:
                crop_data["filtered_clusters"] = [
                    {
                        "cluster_id": c.cluster_id,
                        "points": c.points,  # Include points for visualization
                        "area_mm2": c.area_mm2,
                        "area_cm2": c.area_cm2,
                        "centroid": c.centroid,
                        "bounding_box": c.bounding_box,
                    }
                    for c in r.filtered_clusters
                ]
            crops_data.append(crop_data)
        
        result_data = {
            "filename": path.name,
            "bright_total": bright_total,
            "crops": crops_data,
            "error": None,
            "detector": detector,
        }
        
        # Skip saving annotated images and copies in residual-only mode
        if not residual_only and bright_total > 0:
            noted_dir = output_dir / "Noted"
            images_dir_out = output_dir / "Images"
            
            # Optimize I/O: check existence first to avoid unnecessary work
            noted_path = noted_dir / path.name
            needs_annotation = overwrite or not noted_path.exists()
            
            if needs_annotation:
                # Only now load PIL image for annotation (memory-efficient)
                img, _ = process_image(
                    path=path,
                    crops=crops,
                    threshold=threshold,
                    max_points=max_dots,
                    use_crops=use_crops,
                    load_pil=True,
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
                )
                annotated = annotate_image(
                    img, results, max_dots=max_dots, show_crops=False
                )
                # Use optimized save format (JPEG is faster than PNG for photos)
                # Determine format from original file extension
                save_format = path.suffix.lower()
                if save_format in ('.jpg', '.jpeg'):
                    annotated.save(noted_path, format='JPEG', quality=95, optimize=False)
                elif save_format == '.png':
                    annotated.save(noted_path, format='PNG', optimize=False)
                else:
                    # Default to JPEG for speed
                    noted_path = noted_path.with_suffix('.jpg')
                    annotated.save(noted_path, format='JPEG', quality=95, optimize=False)
                img.close()  # Free memory immediately
            
            # Only copy original if needed (skip if not overwriting and exists)
            raw_copy_path = images_dir_out / path.name
            if overwrite or not raw_copy_path.exists():
                # Use shutil.copy2 for metadata preservation, but it's sequential
                # For large batches, consider async I/O (future optimization)
                shutil.copy2(path, raw_copy_path)
        
        return result_data
    except Exception as exc:
        # Return error information instead of raising
        return {
            "filename": path.name,
            "bright_total": 0,
            "crops": [],
            "error": str(exc),
            "detector": detector,
        }


def main() -> int:
    args = parse_args()

    if args.tune_residual:
        # Interactive parameter tuning mode
        if not args.tune_residual.exists():
            print(f"Error: Image not found: {args.tune_residual}", file=sys.stderr)
            return 1
        
        if not USE_OPENCV:
            print("ERROR: Residual detector requires OpenCV. Install with: pip install opencv-python", file=sys.stderr)
            return 1
        
        # Load crops if available
        crops = None
        crop = None
        try:
            crops = load_crops(args.crops_file)
            if crops and 0 <= args.tune_crop_index < len(crops):
                crop = crops[args.tune_crop_index]
            elif crops:
                crop = crops[0]
                print(f"Warning: Crop index {args.tune_crop_index} out of range, using first crop", file=sys.stderr)
        except Exception:
            pass  # Use default or whole image
        
        tune_residual_params_interactive(
            image_path=args.tune_residual,
            crop=crop,
            crops_file=args.crops_file,
        )
        return 0

    if args.define_crops:
        rects = define_crops_interactive(args.define_crops)
        if not rects:
            print("No rectangles captured.")
            return 0
        print("Captured rectangles (x, y, w, h):")
        print(rects)
        if args.save_crops_json:
            args.save_crops_json.parent.mkdir(parents=True, exist_ok=True)
            with args.save_crops_json.open("w", encoding="utf-8") as f:
                json.dump(rects, f, indent=2)
            print(f"Saved rectangles to {args.save_crops_json}")
        print(
            "Paste these into DEFAULT_CROPS or store them in a JSON file and pass via --crops-file."
        )
        return 0

    try:
        crops = load_crops(args.crops_file)
    except Exception as exc:  # pragma: no cover - simple CLI error path
        print(f"Failed to load crops: {exc}", file=sys.stderr)
        return 1
    
    # Validate residual detector requires OpenCV
    if args.detector == "residual" and not USE_OPENCV:
        print("ERROR: Residual detector requires OpenCV. Install with: pip install opencv-python", file=sys.stderr)
        return 1
    
    try:
        renderer = ProgressRenderer(enable=sys.stdout.isatty())

        if args.batch_size and args.batch_size > 0:
            all_images = list(iter_images(args.images_dir))
            if not all_images:
                raise FileNotFoundError(f"No images found in {args.images_dir}")

            batches = [
                all_images[i : i + args.batch_size] for i in range(0, len(all_images), args.batch_size)
            ]

            print(f"Processing {len(all_images)} images in {len(batches)} batch(es) of up to {args.batch_size} each.")

            all_details: List[Dict[str, object]] = []
            all_found: List[str] = []
            total_processed = 0

            for idx, batch_paths in enumerate(batches, start=1):
                batch_output = args.output_dir / f"batch_{idx:03d}"
                print(f"\nBatch {idx}/{len(batches)} -> {batch_output} ({len(batch_paths)} images)")
                summary = process_batch(
                    images_dir=args.images_dir,
                    output_dir=batch_output,
                    crops=crops,
                    threshold=args.threshold,
                    max_dots=args.max_dots,
                    overwrite=args.overwrite,
                    use_crops=True,  # CLI always uses crops
                    progress_renderer=renderer,
                    image_paths_override=batch_paths,
                    detector=args.detector,
                    resid_blur_ksize=args.resid_blur_ksize,
                    resid_percentile=args.resid_percentile,
                    resid_min=args.resid_min,
                    resid_morph_ksize=args.resid_morph_ksize,
                    min_blob_area_mm2=args.min_blob_area_mm2,
                    debug_residual=args.debug_residual,
                    debug_max=args.debug_max,
                    residual_only=args.residual_only,
                )
                all_details.extend(summary.get("details", []))
                all_found.extend(summary.get("found_filenames", []))
                total_processed += summary.get("processed", 0)

            # Final summary across batches
            print(
                f"\nCompleted {total_processed} images across {len(batches)} batches. "
                f"Detected {len(all_found)} image(s) with bright spots."
            )
            print(f"Per-batch outputs are under {args.output_dir}.")
            return 0

        # No batching: regular single run
        summary = process_batch(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            crops=crops,
            threshold=args.threshold,
            max_dots=args.max_dots,
            overwrite=args.overwrite,
            use_crops=True,  # CLI always uses crops
            progress_renderer=renderer,
            detector=args.detector,
            resid_blur_ksize=args.resid_blur_ksize,
            resid_percentile=args.resid_percentile,
            resid_min=args.resid_min,
            resid_morph_ksize=args.resid_morph_ksize,
            min_blob_area_mm2=args.min_blob_area_mm2,
            debug_residual=args.debug_residual,
            debug_max=args.debug_max,
            residual_only=args.residual_only,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.residual_only:
        debug_dir = args.output_dir / "debug_residual"
        print(
            f"Processed {summary['processed']} image(s). "
            f"Residual debug images saved to {debug_dir}."
        )
    else:
        noted_dir = args.output_dir / "Noted"
        images_dir_out = args.output_dir / "Images"
        print(
            f"Detected {summary['with_bright']} image(s) with bright spots. "
            f"Annotated copies in {noted_dir}, originals in {images_dir_out}, summary.csv in {args.output_dir}."
        )
    return 0


# ----------------------- FastAPI application ----------------------- #
if FastAPI is not None:

    class CropRect(BaseModel):
        x: int
        y: int
        w: int
        h: int

        def to_tuple(self) -> Tuple[int, int, int, int]:
            return (self.x, self.y, self.w, self.h)

    class CropsPayload(BaseModel):
        crops: List[CropRect]

    class DetectRequest(BaseModel):
        image_path: Path = Field(..., description="Absolute or relative path to the image")
        threshold: int = 228
        max_dots: int = 1000
        crops: Optional[List[CropRect]] = None
        use_crops: bool = Field(True, description="If False, check whole image without cropping")
        module_width_mm: Optional[float] = Field(None, description="Module width in millimeters")
        module_height_mm: Optional[float] = Field(None, description="Module height in millimeters")
        bright_spot_area_threshold: Optional[float] = Field(None, description="Minimum area threshold for bright spots (in mm² or cm²)")
        area_unit: str = Field("mm", description="Unit for area threshold: 'mm' for mm² or 'cm' for cm²")
        dbscan_eps_mm: float = Field(2.0, description="DBSCAN distance threshold in mm")
        dbscan_min_samples: int = Field(3, description="DBSCAN minimum samples per cluster")
        detector: str = Field("threshold", description="Detection method: 'threshold' or 'residual'")
        resid_blur_ksize: int = Field(51, description="Gaussian blur kernel size for residual method")
        resid_percentile: float = Field(99.7, description="Percentile for adaptive threshold in residual method")
        resid_min: int = Field(10, description="Minimum residual value floor")
        resid_morph_ksize: int = Field(3, description="Morphology kernel size for cleanup")
        min_blob_area_mm2: Optional[float] = Field(None, description="Minimum blob area in mm²")
        debug_residual: bool = Field(False, description="Enable debug output for residual method")
        debug_max: int = Field(50, description="Maximum number of images to generate debug artifacts for")

    class RunRequest(BaseModel):
        images_dir: str = Field(..., description="Directory containing images (as string path)")
        output_dir: str = Field("output", description="Output directory (as string path)")
        threshold: int = 228
        max_dots: int = 1000
        overwrite: bool = False
        crops: Optional[List[CropRect]] = None
        use_crops: bool = Field(True, description="If False, check whole image without cropping")
        max_workers: Optional[int] = Field(
            None,
            description="Number of parallel workers (None = CPU count, 1 = sequential)",
        )
        module_width_mm: Optional[float] = Field(None, description="Module width in millimeters")
        module_height_mm: Optional[float] = Field(None, description="Module height in millimeters")
        bright_spot_area_threshold: Optional[float] = Field(None, description="Minimum area threshold for bright spots (in mm² or cm²)")
        area_unit: str = Field("mm", description="Unit for area threshold: 'mm' for mm² or 'cm' for cm²")
        dbscan_eps_mm: float = Field(2.0, description="DBSCAN distance threshold in mm")
        dbscan_min_samples: int = Field(3, description="DBSCAN minimum samples per cluster")
        detector: str = Field("threshold", description="Detection method: 'threshold' or 'residual'")
        resid_blur_ksize: int = Field(51, description="Gaussian blur kernel size for residual method")
        resid_percentile: float = Field(99.7, description="Percentile for adaptive threshold in residual method")
        resid_min: int = Field(10, description="Minimum residual value floor")
        resid_morph_ksize: int = Field(3, description="Morphology kernel size for cleanup")
        min_blob_area_mm2: Optional[float] = Field(None, description="Minimum blob area in mm²")
        debug_residual: bool = Field(False, description="Enable debug output for residual method")
        debug_max: int = Field(50, description="Maximum number of images to generate debug artifacts for")
        
        def get_images_dir_path(self) -> Path:
            """Convert images_dir string to Path object, handling Windows paths."""
            # Path() constructor handles both forward and backslashes correctly
            # but we normalize to handle any edge cases
            path_str = str(self.images_dir).replace('\\', os.sep).replace('/', os.sep)
            return Path(path_str)
        
        def get_output_dir_path(self) -> Path:
            """Convert output_dir string to Path object, handling Windows paths."""
            path_str = str(self.output_dir).replace('\\', os.sep).replace('/', os.sep)
            return Path(path_str)

    def create_api_app() -> FastAPI:
        api = FastAPI(title="Bright Spot Detector", version="0.1.0")
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add middleware to log all requests
        from fastapi import Request
        from fastapi.responses import JSONResponse
        from starlette.middleware.base import BaseHTTPMiddleware
        
        class LoggingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                print(f"[REQUEST] {request.method} {request.url.path}", flush=True)
                print(f"[REQUEST] Query params: {request.query_params}", flush=True)
                if request.method in ["POST", "PUT", "PATCH"]:
                    try:
                        body = await request.body()
                        if body:
                            print(f"[REQUEST] Body: {body.decode('utf-8')[:200]}...", flush=True)
                    except:
                        pass
                response = await call_next(request)
                print(f"[REQUEST] Response status: {response.status_code}", flush=True)
                return response
        
        api.add_middleware(LoggingMiddleware)
        
        # Add exception handler for 404 to provide better error messages
        
        @api.exception_handler(404)
        async def not_found_handler(request: Request, exc):
            print(f"[404 HANDLER] ===== 404 Error ======", flush=True)
            print(f"[404 HANDLER] Method: {request.method}", flush=True)
            print(f"[404 HANDLER] URL path: {request.url.path}", flush=True)
            print(f"[404 HANDLER] Full URL: {request.url}", flush=True)
            print(f"[404 HANDLER] Query params: {request.query_params}", flush=True)
            print(f"[404 HANDLER] Path params: {request.path_params}", flush=True)
            print(f"[404 HANDLER] Available routes:", flush=True)
            for route in api.routes:
                if hasattr(route, 'path'):
                    methods = getattr(route, 'methods', set())
                    print(f"[404 HANDLER]   {list(methods)} {route.path}", flush=True)
            print(f"[404 HANDLER] Exception: {exc}", flush=True)
            return JSONResponse(
                status_code=404,
                content={"detail": f"Route not found: {request.method} {request.url.path}. Check server logs for available routes."}
            )

        @api.get("/api/health")
        def health() -> Dict[str, str]:
            return {"status": "ok"}
        
        @api.post("/api/run/test")
        def test_run_endpoint(payload: RunRequest) -> Dict[str, object]:
            """Test endpoint to verify request parsing."""
            return {
                "received": True,
                "images_dir": payload.images_dir,
                "images_dir_type": type(payload.images_dir).__name__,
                "path_exists": payload.get_images_dir_path().exists(),
                "path": str(payload.get_images_dir_path()),
            }

        @api.get("/api/images/{subdir}/{filename:path}")
        def serve_image(subdir: str, filename: str, output_dir: Optional[str] = None):
            """Serve images from the output directory via HTTP.
            
            Query param output_dir is required and should be the output directory path.
            """
            # Security: only allow "Noted" or "Images" subdirectories
            if subdir not in ("Noted", "Images"):
                raise HTTPException(status_code=403, detail="Invalid subdirectory")
            if not output_dir:
                raise HTTPException(status_code=400, detail="output_dir query parameter required")
            image_path = Path(output_dir) / subdir / filename
            if not image_path.exists():
                raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
            return FileResponse(image_path)

        @api.get("/api/crops")
        def get_crops(crops_file: Path = Path("crops.json")) -> Dict[str, object]:
            try:
                crops = load_crops(crops_file if crops_file.exists() else None)
            except Exception as exc:  # pragma: no cover - runtime error path
                raise HTTPException(status_code=400, detail=str(exc))
            return {"crops": [list(r) for r in crops], "source": str(crops_file)}

        @api.post("/api/crops")
        def set_crops(payload: CropsPayload, crops_file: Path = Path("crops.json")):
            save_crops([c.to_tuple() for c in payload.crops], crops_file)
            return {"saved": True, "crops_file": str(crops_file)}

        @api.post("/api/detect")
        def detect(payload: DetectRequest) -> Dict[str, object]:
            if not payload.image_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"Image not found: {payload.image_path}"
                )
            try:
                crops = (
                    [c.to_tuple() for c in payload.crops]
                    if payload.crops is not None
                    else load_crops(Path("crops.json") if Path("crops.json").exists() else None)
                )
                result = detect_single(
                    path=payload.image_path,
                    crops=crops,
                    threshold=payload.threshold,
                    max_dots=payload.max_dots,
                    output_dir=Path("output") / "live",
                    use_crops=payload.use_crops,
                    module_width_mm=payload.module_width_mm,
                    module_height_mm=payload.module_height_mm,
                    bright_spot_area_threshold=payload.bright_spot_area_threshold,
                    area_unit=payload.area_unit,
                    dbscan_eps_mm=payload.dbscan_eps_mm,
                    dbscan_min_samples=payload.dbscan_min_samples,
                    detector=payload.detector,
                    resid_blur_ksize=payload.resid_blur_ksize,
                    resid_percentile=payload.resid_percentile,
                    resid_min=payload.resid_min,
                    resid_morph_ksize=payload.resid_morph_ksize,
                    min_blob_area_mm2=payload.min_blob_area_mm2,
                    debug_residual=payload.debug_residual,
                    debug_max=payload.debug_max,
                )
            except Exception as exc:  # pragma: no cover
                raise HTTPException(status_code=400, detail=str(exc))
            return result

        def _run_job(job_id: str, req: RunRequest) -> None:
            import traceback
            print(f"[JOB {job_id}] ===== Background job started =====", flush=True)
            print(f"[JOB {job_id}] Request details: images_dir='{req.images_dir}', output_dir='{req.output_dir}'", flush=True)
            
            PROGRESS[job_id] = {
                "status": "running",
                "current": 0,
                "total": 0,
                "elapsed": 0.0,
                "estimated_remaining": None,
            }
            print(f"[JOB {job_id}] Updated PROGRESS with 'running' status", flush=True)
            print(f"[JOB {job_id}] PROGRESS keys: {list(PROGRESS.keys())}", flush=True)
            
            try:
                print(f"[JOB {job_id}] Loading crops...", flush=True)
                crops = (
                    [c.to_tuple() for c in req.crops]
                    if req.crops is not None
                    else load_crops(Path("crops.json") if Path("crops.json").exists() else None)
                )
                print(f"[JOB {job_id}] Loaded {len(crops) if crops else 0} crops, use_crops={req.use_crops}", flush=True)
                
                images_dir = req.get_images_dir_path()
                output_dir = req.get_output_dir_path()
                print(f"[JOB {job_id}] Images dir: {images_dir}", flush=True)
                print(f"[JOB {job_id}] Output dir: {output_dir}", flush=True)
                
                print(f"[JOB {job_id}] Scanning for images...", flush=True)
                images = list(iter_images(images_dir))
                print(f"[JOB {job_id}] Found {len(images)} images", flush=True)
                PROGRESS[job_id]["total"] = len(images)
                print(f"[JOB {job_id}] Updated PROGRESS total to {len(images)}", flush=True)
                
                def progress_callback(current: int, total: int, elapsed: float, estimated_remaining: Optional[float]) -> None:
                    PROGRESS[job_id].update(
                        current=current,
                        total=total,
                        elapsed=elapsed,
                        estimated_remaining=estimated_remaining,
                    )
                    # Send SSE event to all connected streams
                    # Note: STREAMS access is not fully thread-safe, but queue.put() is atomic
                    # For production, consider using a proper lock
                    if job_id in STREAMS:
                        event_data = {
                            "current": current,
                            "total": total,
                            "elapsed": elapsed,
                            "estimated_remaining": estimated_remaining,
                            "status": "running",
                        }
                        for stream in STREAMS[job_id]:
                            try:
                                stream.put(event_data)
                            except Exception as e:
                                print(f"[JOB {job_id}] WARNING: Failed to send progress to stream: {e}", flush=True)
                
                print(f"[JOB {job_id}] Starting process_batch...", flush=True)
                summary = process_batch(
                    images_dir=images_dir,
                    output_dir=output_dir,
                    crops=crops,
                    threshold=req.threshold,
                    max_dots=req.max_dots,
                    overwrite=req.overwrite,
                    progress_cb=progress_callback,
                    max_workers=req.max_workers,
                    use_crops=req.use_crops,
                    module_width_mm=req.module_width_mm,
                    module_height_mm=req.module_height_mm,
                    bright_spot_area_threshold=req.bright_spot_area_threshold,
                    area_unit=req.area_unit,
                    dbscan_eps_mm=req.dbscan_eps_mm,
                    dbscan_min_samples=req.dbscan_min_samples,
                    detector=req.detector,
                    resid_blur_ksize=req.resid_blur_ksize,
                    resid_percentile=req.resid_percentile,
                    resid_min=req.resid_min,
                    resid_morph_ksize=req.resid_morph_ksize,
                    min_blob_area_mm2=req.min_blob_area_mm2,
                    debug_residual=req.debug_residual,
                    debug_max=req.debug_max,
                )
                print(f"[JOB {job_id}] process_batch completed. Processed: {summary.get('processed', 0)}", flush=True)
                
                PROGRESS[job_id].update(
                    status="completed",
                    summary=summary,
                    current=summary["processed"],
                )
                print(f"[JOB {job_id}] Updated PROGRESS with 'completed' status", flush=True)
                
                # Send completion event
                if job_id in STREAMS:
                    event_data = {
                        "status": "completed",
                        "summary": summary,
                        "current": summary["processed"],
                        "total": summary["processed"],
                    }
                    print(f"[JOB {job_id}] Sending completion event to {len(STREAMS[job_id])} stream(s)", flush=True)
                    for stream in STREAMS[job_id]:
                        try:
                            stream.put(event_data)
                        except Exception as e:
                            print(f"[JOB {job_id}] WARNING: Failed to send completion to stream: {e}", flush=True)
                else:
                    print(f"[JOB {job_id}] No streams registered for this job", flush=True)
                    
            except Exception as exc:  # pragma: no cover - runtime path
                error_detail = str(exc)
                error_traceback = traceback.format_exc()
                print(f"[JOB {job_id}] ERROR: Exception occurred: {error_detail}", flush=True)
                print(f"[JOB {job_id}] ERROR Traceback:\n{error_traceback}", flush=True)
                
                PROGRESS[job_id] = {"status": "error", "detail": error_detail}
                print(f"[JOB {job_id}] Updated PROGRESS with 'error' status", flush=True)
                print(f"[JOB {job_id}] PROGRESS after error: {PROGRESS.get(job_id)}", flush=True)
                
                # Send error event
                if job_id in STREAMS:
                    event_data = {"status": "error", "detail": error_detail}
                    print(f"[JOB {job_id}] Sending error event to {len(STREAMS[job_id])} stream(s)", flush=True)
                    for stream in STREAMS[job_id]:
                        try:
                            stream.put(event_data)
                        except Exception as e:
                            print(f"[JOB {job_id}] WARNING: Failed to send error to stream: {e}", flush=True)
                else:
                    print(f"[JOB {job_id}] No streams registered for this job, cannot send error event", flush=True)

        @api.post("/api/run")
        def run_batch_endpoint(
            payload: RunRequest, background: BackgroundTasks
        ) -> Dict[str, str]:
            import traceback
            print(f"[RUN] ===== Run batch endpoint called =====", flush=True)
            print(f"[RUN] Request payload: images_dir='{payload.images_dir}', output_dir='{payload.output_dir}'", flush=True)
            print(f"[RUN] Threshold: {payload.threshold}, max_dots: {payload.max_dots}, overwrite: {payload.overwrite}", flush=True)
            print(f"[RUN] max_workers: {payload.max_workers}", flush=True)
            
            try:
                # Convert string path to Path object
                images_dir = payload.get_images_dir_path()
                print(f"[RUN] Converted images_dir to Path: {images_dir}", flush=True)
                
                # Resolve the path to handle relative paths and symlinks
                try:
                    images_dir = images_dir.resolve()
                    print(f"[RUN] Resolved images_dir to: {images_dir}", flush=True)
                except (OSError, RuntimeError) as e:
                    # If resolve fails, use the path as-is
                    print(f"[RUN] WARNING: Could not resolve path: {e}", flush=True)
                    pass
                
                if not images_dir.exists():
                    error_msg = f"Images directory not found: {images_dir} (resolved from: {payload.images_dir})"
                    print(f"[RUN] ERROR: {error_msg}", flush=True)
                    raise HTTPException(status_code=404, detail=error_msg)
                if not images_dir.is_dir():
                    error_msg = f"Path is not a directory: {images_dir}"
                    print(f"[RUN] ERROR: {error_msg}", flush=True)
                    raise HTTPException(status_code=400, detail=error_msg)
                
                print(f"[RUN] Images directory validated successfully: {images_dir}", flush=True)
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                print(f"[RUN] HTTPException raised, re-raising...", flush=True)
                raise
            except (TypeError, ValueError, OSError) as e:
                error_msg = f"Invalid images_dir path: {payload.images_dir}. Error: {str(e)}"
                print(f"[RUN] ERROR: {error_msg}", flush=True)
                print(f"[RUN] Traceback: {traceback.format_exc()}", flush=True)
                raise HTTPException(status_code=400, detail=error_msg)
            except Exception as e:
                # Catch any other unexpected errors
                error_msg = f"Unexpected error processing request: {str(e)}"
                print(f"[RUN] ERROR: {error_msg}", flush=True)
                print(f"[RUN] Traceback: {traceback.format_exc()}", flush=True)
                raise HTTPException(status_code=500, detail=error_msg)
            
            job_id = uuid.uuid4().hex
            PROGRESS[job_id] = {"status": "queued", "current": 0, "total": 0}
            print(f"[RUN] Created job_id='{job_id}', added to PROGRESS. Total jobs now: {len(PROGRESS)}", flush=True)
            print(f"[RUN] PROGRESS keys after creation: {list(PROGRESS.keys())}", flush=True)
            print(f"[RUN] Scheduling background task for job_id='{job_id}'...", flush=True)
            background.add_task(_run_job, job_id, payload)
            print(f"[RUN] Background task scheduled. Returning job_id='{job_id}'", flush=True)
            return {"job_id": job_id}

        # IMPORTANT: More specific routes must be defined BEFORE more general ones
        # Otherwise FastAPI will match /api/run/{job_id}/stream to /api/run/{job_id} first
        @api.get("/api/run/{job_id}/stream")
        def stream_run_progress(job_id: str, request: Request):
            """Stream progress updates via Server-Sent Events (SSE)."""
            import traceback
            
            # Debug: print what we received
            print(f"[STREAM] ===== Stream endpoint called =====", flush=True)
            print(f"[STREAM] job_id='{job_id}', type={type(job_id)}", flush=True)
            print(f"[STREAM] Request URL: {request.url if request else 'N/A'}", flush=True)
            print(f"[STREAM] Request method: {request.method if request else 'N/A'}", flush=True)
            print(f"[STREAM] Request path: {request.url.path if request else 'N/A'}", flush=True)
            print(f"[STREAM] PROGRESS keys: {list(PROGRESS.keys())}", flush=True)
            print(f"[STREAM] PROGRESS contents: {PROGRESS}", flush=True)
            print(f"[STREAM] STREAMS keys: {list(STREAMS.keys())}", flush=True)
            
            # Normalize job_id (remove any whitespace)
            original_job_id = job_id
            job_id = job_id.strip()
            if original_job_id != job_id:
                print(f"[STREAM] WARNING: job_id had whitespace. Original: '{original_job_id}', Normalized: '{job_id}'", flush=True)
            
            # Wait a bit for the job to be initialized (handles race condition)
            import time
            max_wait = 5  # Wait up to 5 seconds for job to appear
            wait_interval = 0.1
            waited = 0.0
            print(f"[STREAM] Waiting for job '{job_id}' to appear in PROGRESS...", flush=True)
            while job_id not in PROGRESS and waited < max_wait:
                time.sleep(wait_interval)
                waited += wait_interval
                if waited % 1.0 < wait_interval:  # Print every second
                    print(f"[STREAM] Still waiting... ({waited:.1f}s / {max_wait}s)", flush=True)
            
            if job_id not in PROGRESS:
                available = list(PROGRESS.keys())[:5] if PROGRESS else []
                error_msg = f"Job '{job_id}' not found. Available jobs: {available}. Total jobs: {len(PROGRESS)}"
                print(f"[STREAM] ERROR: {error_msg}", flush=True)
                print(f"[STREAM] Full PROGRESS dict: {PROGRESS}", flush=True)
                print(f"[STREAM] Full STREAMS dict: {STREAMS}", flush=True)
                raise HTTPException(status_code=404, detail=error_msg)
            
            print(f"[STREAM] Job '{job_id}' found in PROGRESS. Status: {PROGRESS[job_id].get('status', 'unknown')}", flush=True)
            
            import queue
            
            event_queue: queue.Queue = queue.Queue()
            
            # Add this stream to the job's stream list
            # Use a lock for thread safety
            if job_id not in STREAMS:
                STREAMS[job_id] = []
            STREAMS[job_id].append(event_queue)
            
            def generate():
                import traceback
                try:
                    print(f"[STREAM {job_id}] Generator started", flush=True)
                    # Send initial state
                    initial_data = PROGRESS.get(job_id, {})
                    print(f"[STREAM {job_id}] Sending initial data: {initial_data}", flush=True)
                    yield f"data: {json.dumps(initial_data)}\n\n"
                    
                    # Stream updates
                    iteration = 0
                    while True:
                        iteration += 1
                        try:
                            event_data = event_queue.get(timeout=1.0)
                            print(f"[STREAM {job_id}] Received event data (iteration {iteration}): {event_data}", flush=True)
                            yield f"data: {json.dumps(event_data)}\n\n"
                            
                            # Stop streaming if job is completed or errored
                            if event_data.get("status") in ("completed", "error"):
                                print(f"[STREAM {job_id}] Job finished with status: {event_data.get('status')}", flush=True)
                                break
                        except queue.Empty:
                            # Send heartbeat to keep connection alive
                            current_data = PROGRESS.get(job_id, {})
                            if current_data.get("status") in ("completed", "error"):
                                print(f"[STREAM {job_id}] Job finished (from heartbeat check): {current_data.get('status')}", flush=True)
                                yield f"data: {json.dumps(current_data)}\n\n"
                                break
                            # Only log heartbeat every 10 iterations to avoid spam
                            if iteration % 10 == 0:
                                print(f"[STREAM {job_id}] Heartbeat (iteration {iteration}), status: {current_data.get('status', 'unknown')}", flush=True)
                            continue
                        except Exception as e:
                            error_msg = f"Error in stream generator: {str(e)}"
                            print(f"[STREAM {job_id}] ERROR: {error_msg}", flush=True)
                            print(f"[STREAM {job_id}] Traceback:\n{traceback.format_exc()}", flush=True)
                            error_data = {"status": "error", "detail": error_msg}
                            yield f"data: {json.dumps(error_data)}\n\n"
                            break
                except Exception as e:
                    error_msg = f"Fatal error in stream generator: {str(e)}"
                    print(f"[STREAM {job_id}] FATAL ERROR: {error_msg}", flush=True)
                    print(f"[STREAM {job_id}] Traceback:\n{traceback.format_exc()}", flush=True)
                finally:
                    # Clean up stream
                    print(f"[STREAM {job_id}] Cleaning up stream...", flush=True)
                    if job_id in STREAMS and event_queue in STREAMS[job_id]:
                        STREAMS[job_id].remove(event_queue)
                        print(f"[STREAM {job_id}] Removed event_queue from STREAMS", flush=True)
                    if job_id in STREAMS and not STREAMS[job_id]:
                        del STREAMS[job_id]
                        print(f"[STREAM {job_id}] Removed job_id from STREAMS (no more streams)", flush=True)
                    print(f"[STREAM {job_id}] Cleanup complete", flush=True)
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                },
            )

        @api.get("/api/run/{job_id}")
        def get_run_status(job_id: str) -> Dict[str, object]:
            """Get the status of a run job."""
            print(f"[STATUS] Request for job_id='{job_id}'", flush=True)
            print(f"[STATUS] PROGRESS keys: {list(PROGRESS.keys())}", flush=True)
            data = PROGRESS.get(job_id)
            if data is None:
                available = list(PROGRESS.keys())[:5]
                error_msg = f"Job not found: {job_id}. Available jobs: {available}"
                print(f"[STATUS] ERROR: {error_msg}", flush=True)
                raise HTTPException(status_code=404, detail=error_msg)
            print(f"[STATUS] Found job '{job_id}' with status: {data.get('status', 'unknown')}", flush=True)
            return data

        return api

    app = create_api_app()
else:
    app = None


if __name__ == "__main__":
    # Check if user wants to run the API server
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Run the API server
        if app is None:
            print("ERROR: FastAPI is not available. Install with: pip install fastapi uvicorn", file=sys.stderr)
            raise SystemExit(1)
        import uvicorn
        print("Starting FastAPI server on http://127.0.0.1:8000")
        print("API docs available at http://127.0.0.1:8000/docs")
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    else:
        # Run CLI mode
        raise SystemExit(main())