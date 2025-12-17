"""Command-line interface for bright spot detection."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from batch import process_batch
from config import load_crops, save_crops
from image_io import USE_OPENCV, iter_images
from interactive import define_crops_interactive, tune_residual_params_interactive
from progress import ProgressRenderer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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


def main() -> int:
    """Main entry point for CLI."""
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


if __name__ == "__main__":
    raise SystemExit(main())

