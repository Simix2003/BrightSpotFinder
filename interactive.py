"""Interactive tools for parameter tuning and crop definition."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from config import load_crops
from image_io import USE_OPENCV, load_image_optimized

# Import cv2 if available
if USE_OPENCV:
    import cv2
else:
    cv2 = None


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
    """Interactive tool to define crop regions by drawing rectangles on an image.
    
    Args:
        image_path: Path to the image to draw on
    
    Returns:
        List of crop rectangles (x, y, w, h)
    """
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

