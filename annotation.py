"""Image annotation functions for bright spot detection."""

from typing import Sequence

from PIL import Image, ImageDraw

from models import CropResult


def annotate_image(
    img: Image.Image,
    results: Sequence[CropResult],
    max_dots: int,
    show_crops: bool = True,
) -> Image.Image:
    """Annotate image with bright spot detections.
    
    Args:
        img: PIL Image to annotate
        results: Sequence of CropResult objects
        max_dots: Maximum number of dots to draw
        show_crops: Whether to show crop rectangles
    
    Returns:
        Annotated PIL Image
    """
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

