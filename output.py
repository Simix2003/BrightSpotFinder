"""Output generation functions for bright spot detection."""

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


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

