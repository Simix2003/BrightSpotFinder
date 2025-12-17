"""Batch processing functions for bright spot detection."""

import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from annotation import annotate_image
from image_io import ensure_output, iter_images
from output import write_csv
from processing import process_image
from progress import ProgressRenderer


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

