"""Configuration and constants for bright spot detection."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# Rectangles are defined as (x, y, width, height).
DEFAULT_CROPS: List[Tuple[int, int, int, int]] = [
    # Example crops; adjust to your real regions of interest.
    (50, 50, 200, 200),
    (300, 200, 180, 180),
]

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


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

