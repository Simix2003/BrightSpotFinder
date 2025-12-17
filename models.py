"""Data models for bright spot detection."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


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
    """Result of bright spot detection on a crop region."""
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

