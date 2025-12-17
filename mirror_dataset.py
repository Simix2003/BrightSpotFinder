"""
Dataset Mirroring Augmentation Script

This script creates augmented versions of images and labels by applying:
- Horizontal flip (left-right mirror)
- Vertical flip (upside down)
- Both flips combined (180° rotation)

The script correctly transforms YOLO format bounding box coordinates for each flip type.
"""

import cv2
import argparse
from pathlib import Path


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def transform_yolo_label_horizontal(cx, cy, w, h):
    """Transform YOLO coordinates for horizontal flip (left-right mirror)."""
    return (1.0 - cx, cy, w, h)


def transform_yolo_label_vertical(cx, cy, w, h):
    """Transform YOLO coordinates for vertical flip (upside down)."""
    return (cx, 1.0 - cy, w, h)


def transform_yolo_label_both(cx, cy, w, h):
    """Transform YOLO coordinates for both flips (horizontal + vertical)."""
    return (1.0 - cx, 1.0 - cy, w, h)


def load_yolo_labels(label_path):
    """Load YOLO format labels from a text file."""
    if not label_path.exists():
        return []
    
    labels = []
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    
    for line in content.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            labels.append((class_id, cx, cy, w, h))
        except ValueError:
            continue
    
    return labels


def save_yolo_labels(label_path, labels):
    """Save YOLO format labels to a text file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not labels:
        label_path.write_text("", encoding="utf-8")
        return
    
    lines = []
    for class_id, cx, cy, w, h in labels:
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_image_and_labels(image_path, label_path, images_dir, labels_dir, split_name):
    """Process a single image and its labels, creating 3 augmented versions."""
    print(f"Processing: {image_path.name}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [WARN] Failed to load image: {image_path}")
        return
    
    # Load labels
    labels = load_yolo_labels(label_path)
    
    # Get base name without extension
    base_name = image_path.stem
    ext = image_path.suffix
    
    # 1. Horizontal flip
    img_hflip = cv2.flip(img, 1)
    hflip_img_path = images_dir / f"{base_name}_hflip{ext}"
    cv2.imwrite(str(hflip_img_path), img_hflip)
    
    hflip_labels = []
    for class_id, cx, cy, w, h in labels:
        cx_new, cy_new, w_new, h_new = transform_yolo_label_horizontal(cx, cy, w, h)
        hflip_labels.append((class_id, cx_new, cy_new, w_new, h_new))
    
    hflip_label_path = labels_dir / f"{base_name}_hflip.txt"
    save_yolo_labels(hflip_label_path, hflip_labels)
    
    # 2. Vertical flip
    img_vflip = cv2.flip(img, 0)
    vflip_img_path = images_dir / f"{base_name}_vflip{ext}"
    cv2.imwrite(str(vflip_img_path), img_vflip)
    
    vflip_labels = []
    for class_id, cx, cy, w, h in labels:
        cx_new, cy_new, w_new, h_new = transform_yolo_label_vertical(cx, cy, w, h)
        vflip_labels.append((class_id, cx_new, cy_new, w_new, h_new))
    
    vflip_label_path = labels_dir / f"{base_name}_vflip.txt"
    save_yolo_labels(vflip_label_path, vflip_labels)
    
    # 3. Both flips (horizontal + vertical)
    img_hvflip = cv2.flip(img, -1)
    hvflip_img_path = images_dir / f"{base_name}_hvflip{ext}"
    cv2.imwrite(str(hvflip_img_path), img_hvflip)
    
    hvflip_labels = []
    for class_id, cx, cy, w, h in labels:
        cx_new, cy_new, w_new, h_new = transform_yolo_label_both(cx, cy, w, h)
        hvflip_labels.append((class_id, cx_new, cy_new, w_new, h_new))
    
    hvflip_label_path = labels_dir / f"{base_name}_hvflip.txt"
    save_yolo_labels(hvflip_label_path, hvflip_labels)
    
    print(f"  Created: {base_name}_hflip, {base_name}_vflip, {base_name}_hvflip")


def process_split(dataset_path, split_name):
    """Process all images in a split (train or val)."""
    images_dir = dataset_path / "images" / split_name
    labels_dir = dataset_path / "labels" / split_name
    
    if not images_dir.exists():
        print(f"[WARN] Images directory not found: {images_dir}")
        return 0
    
    # Find all images
    image_files = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    
    # Filter out already augmented images (those with _hflip, _vflip, _hvflip suffixes)
    original_images = [
        img for img in image_files
        if not any(suffix in img.stem for suffix in ["_hflip", "_vflip", "_hvflip"])
    ]
    
    if not original_images:
        print(f"[INFO] No original images found in {split_name} (may already be processed)")
        return 0
    
    print(f"\n[{split_name.upper()}] Processing {len(original_images)} images...")
    
    processed = 0
    for img_path in original_images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        process_image_and_labels(img_path, label_path, images_dir, labels_dir, split_name)
        processed += 1
    
    print(f"[{split_name.upper()}] Processed {processed} images")
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Mirror dataset images and transform YOLO labels"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to YOLO dataset folder (containing images/, labels/, data.yaml)"
    )
    
    args = parser.parse_args()
    
    dataset_path = args.dataset.resolve()
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        return
    
    print(f"Dataset path: {dataset_path}")
    print("Creating augmented versions (horizontal flip, vertical flip, both flips)...")
    print("This will quadruple your dataset size.\n")
    
    # Process train and val splits
    train_count = process_split(dataset_path, "train")
    val_count = process_split(dataset_path, "val")
    
    total_original = train_count + val_count
    total_augmented = total_original * 3  # 3 augmented versions per image
    
    print(f"\n[DONE]")
    print(f"  Original images: {total_original}")
    print(f"  Augmented images created: {total_augmented}")
    print(f"  Total images now: {total_original + total_augmented} (4x original)")


if __name__ == "__main__":
    main()

