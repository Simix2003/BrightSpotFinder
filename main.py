import cv2
import numpy as np
from pathlib import Path

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def compute_residual(
    gray: np.ndarray,
    blur_ksize: int = 75,
) -> np.ndarray:
    """
    Compute residual image:
    residual = gray - local_background(gray)
    
    Keeps only local bright stuff relative to the blurred background.
    Returns uint8 image (0–255).
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    # Background estimation
    background = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Residual (keeps only local bright stuff)
    residual = cv2.subtract(gray, background)

    return residual


def normalize_percentile(residual: np.ndarray, lo: float = 1.0, hi: float = 99.7) -> np.ndarray:
    """
    Robust normalization: ignore extreme values (grid edges, borders).
    Uses percentile-based scaling to avoid being dominated by outliers.
    """
    a = residual.astype(np.float32)
    p_lo, p_hi = np.percentile(a, (lo, hi))
    if p_hi <= p_lo + 1e-6:
        return np.zeros_like(residual, dtype=np.uint8)

    out = (a - p_lo) * 255.0 / (p_hi - p_lo)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def process_image(
    img_path: Path,
    output_dir: Path,
    blur_ksize: int,
    use_tophat: bool = False,
    tophat_kernel_size: int = 9,
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.7,
):
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[WARN] Could not read {img_path}")
        return

    # Compute raw residual
    res_raw = compute_residual(gray, blur_ksize=blur_ksize)
    
    # Optionally apply top-hat to emphasize small bright objects
    if use_tophat:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_kernel_size, tophat_kernel_size))
        res_raw = cv2.morphologyEx(res_raw, cv2.MORPH_TOPHAT, kernel)
    
    # Normalize for visualization using percentile-based robust scaling
    #res_vis = normalize_percentile(res_raw, lo=percentile_lo, hi=percentile_hi)

    # Save both versions
    base_name = img_path.stem
    ext = img_path.suffix
    
    out_path_raw = output_dir / f"{base_name}_residual_raw{ext}"
    #out_path_vis = output_dir / f"{base_name}_residual_vis{ext}"
    
    cv2.imwrite(str(out_path_raw), res_raw)
    #cv2.imwrite(str(out_path_vis), res_vis)


def generate_residuals(
    input_path: Path,
    output_dir: Path,
    blur_ksize: int = 75,
    use_tophat: bool = False,
    tophat_kernel_size: int = 9,
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.7,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        process_image(
            input_path, output_dir, blur_ksize,
            use_tophat=use_tophat,
            tophat_kernel_size=tophat_kernel_size,
            percentile_lo=percentile_lo,
            percentile_hi=percentile_hi,
        )
        print(f"[OK] Residual saved for {input_path.name}")

    elif input_path.is_dir():
        images = [
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]

        print(f"[INFO] Found {len(images)} images")

        for i, img_path in enumerate(images, 1):
            process_image(
                img_path, output_dir, blur_ksize,
                use_tophat=use_tophat,
                tophat_kernel_size=tophat_kernel_size,
                percentile_lo=percentile_lo,
                percentile_hi=percentile_hi,
            )
            if i % 50 == 0 or i == len(images):
                print(f"[PROGRESS] {i}/{len(images)}")

        print(f"[DONE] Residuals saved to {output_dir}")

    else:
        raise ValueError("Input path is neither a file nor a directory")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate residual images")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input image or folder")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output folder for residual images")
    parser.add_argument("--blur-ksize", type=int, default=75,
                        help="Gaussian blur kernel size (odd, large = smoother background)")
    parser.add_argument("--use-tophat", action="store_true",
                        help="Apply top-hat morphology to emphasize small bright objects")
    parser.add_argument("--tophat-kernel-size", type=int, default=9,
                        help="Top-hat kernel size (only used if --use-tophat is set)")
    parser.add_argument("--percentile-lo", type=float, default=1.0,
                        help="Lower percentile for robust normalization (default: 1.0)")
    parser.add_argument("--percentile-hi", type=float, default=99.7,
                        help="Upper percentile for robust normalization (default: 99.7)")

    args = parser.parse_args()

    generate_residuals(
        input_path=args.input,
        output_dir=args.output,
        blur_ksize=args.blur_ksize,
        use_tophat=args.use_tophat,
        tophat_kernel_size=args.tophat_kernel_size,
        percentile_lo=args.percentile_lo,
        percentile_hi=args.percentile_hi,
    )
