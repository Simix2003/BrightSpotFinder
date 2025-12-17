"""
Migration Script: Convert YOLO text annotations to annotated images

This script reads YOLO format text files from the Noted folder,
draws bounding boxes on corresponding images from the Image folder,
saves annotated images back to Noted folder, and deletes the text files.
"""

import cv2
import os
from pathlib import Path


def load_yolo_labels(label_path, img_width, img_height):
    """Load YOLO format labels and convert to pixel coordinates."""
    if not label_path.exists():
        return []
    
    boxes = []
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
            
            # Convert normalized coordinates to pixel coordinates
            cx_px = cx * img_width
            cy_px = cy * img_height
            w_px = w * img_width
            h_px = h * img_height
            
            # Calculate bounding box corners
            x1 = int(round(cx_px - w_px / 2.0))
            y1 = int(round(cy_px - h_px / 2.0))
            x2 = int(round(cx_px + w_px / 2.0))
            y2 = int(round(cy_px + h_px / 2.0))
            
            # Clamp to image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            boxes.append((class_id, x1, y1, x2, y2))
        except ValueError:
            continue
    
    return boxes


def draw_boxes_on_image(img, boxes):
    """Draw bounding boxes on image."""
    annotated = img.copy()
    
    for class_id, x1, y1, x2, y2 in boxes:
        # Draw rectangle in green
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"bright_spot"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1, label_size[1] + 10)
        
        # Draw label background
        cv2.rectangle(
            annotated,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0], label_y + 5),
            (0, 255, 0),
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    return annotated


def migrate_annotations(noted_dir, images_dir):
    """Convert text annotations to annotated images."""
    noted_path = Path(noted_dir)
    images_path = Path(images_dir)
    
    if not noted_path.exists():
        print(f"[ERROR] Noted directory does not exist: {noted_dir}")
        return
    
    if not images_path.exists():
        print(f"[ERROR] Images directory does not exist: {images_dir}")
        return
    
    # Find all text files in Noted folder
    text_files = list(noted_path.glob("*.txt"))
    
    if not text_files:
        print("[INFO] No text annotation files found in Noted folder")
        return
    
    print(f"Found {len(text_files)} annotation files to process...\n")
    
    processed = 0
    skipped = 0
    
    for txt_file in text_files:
        # Get corresponding image name (same stem, try common extensions)
        stem = txt_file.stem
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        img_file = None
        
        for ext in image_extensions:
            candidate = images_path / f"{stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file is None:
            print(f"[WARN] Image not found for {txt_file.name}, skipping...")
            skipped += 1
            continue
        
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"[WARN] Failed to load image: {img_file}, skipping...")
            skipped += 1
            continue
        
        h, w = img.shape[:2]
        
        # Load annotations
        boxes = load_yolo_labels(txt_file, w, h)
        
        if not boxes:
            print(f"[INFO] No boxes found in {txt_file.name}, creating empty annotated image...")
        
        # Draw boxes on image
        annotated_img = draw_boxes_on_image(img, boxes)
        
        # Save annotated image (same name as text file but with image extension)
        output_ext = img_file.suffix
        annotated_path = noted_path / f"{stem}{output_ext}"
        cv2.imwrite(str(annotated_path), annotated_img)
        
        # Delete text file
        txt_file.unlink()
        
        processed += 1
        print(f"[✓] Processed: {txt_file.name} -> {annotated_path.name}")
    
    print(f"\n{'='*60}")
    print("MIGRATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total annotation files found: {len(text_files)}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped (image not found): {skipped}")
    print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate YOLO text annotations to annotated images"
    )
    parser.add_argument(
        "--noted-dir",
        type=str,
        required=True,
        help="Path to Noted folder containing text annotation files"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Path to Image folder containing original images"
    )
    
    args = parser.parse_args()
    
    migrate_annotations(args.noted_dir, args.images_dir)


if __name__ == "__main__":
    main()

