"""
Dataset Visualization Script

This script displays images from a YOLO dataset with bounding boxes drawn
from the corresponding label files. Allows navigation through images one by one.
"""

import cv2
import argparse
from pathlib import Path


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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


def get_class_name(class_id, data_yaml_path):
    """Get class name from data.yaml if available."""
    try:
        import yaml
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                names = data.get('names', {})
                if isinstance(names, dict):
                    return names.get(class_id, f"class_{class_id}")
                elif isinstance(names, list):
                    if class_id < len(names):
                        return names[class_id]
    except Exception:
        pass
    return f"class_{class_id}"


def draw_boxes(img, boxes, class_names_func):
    """Draw bounding boxes on image."""
    vis = img.copy()
    
    for class_id, x1, y1, x2, y2 in boxes:
        # Draw rectangle in green
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get class name
        class_name = class_names_func(class_id)
        
        # Draw class label
        label = f"{class_id}:{class_name}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1, label_size[1] + 10)
        
        # Draw label background
        cv2.rectangle(
            vis,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0], label_y + 5),
            (0, 255, 0),
            -1
        )
        
        # Draw label text
        cv2.putText(
            vis,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    return vis


def collect_images(dataset_path):
    """Collect all images from train and val splits."""
    images = []
    
    for split in ["train", "val"]:
        images_dir = dataset_path / "images" / split
        labels_dir = dataset_path / "labels" / split
        
        if not images_dir.exists():
            continue
        
        image_files = [
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
        image_files.sort()
        
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            images.append((img_path, label_path, split))
    
    return images


def visualize_dataset(dataset_path):
    """Main visualization function."""
    dataset_path = dataset_path.resolve()
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        return
    
    # Collect all images
    images = collect_images(dataset_path)
    
    if not images:
        print("[ERROR] No images found in dataset")
        return
    
    print(f"Found {len(images)} images")
    print("\nControls:")
    print("  [n] or [→] : Next image")
    print("  [p] or [←] : Previous image")
    print("  [q] or [ESC] : Quit")
    print("\nPress any key to start...")
    
    # Load class names
    data_yaml_path = dataset_path / "data.yaml"
    class_names_func = lambda cid: get_class_name(cid, data_yaml_path)
    
    # Create window
    win_name = "YOLO Dataset Visualizer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    idx = 0
    
    while True:
        img_path, label_path, split = images[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to load: {img_path}")
            idx = (idx + 1) % len(images)
            continue
        
        h, w = img.shape[:2]
        
        # Load and draw boxes
        boxes = load_yolo_labels(label_path, w, h)
        vis = draw_boxes(img, boxes, class_names_func)
        
        # Add status bar
        status = f"[{idx+1}/{len(images)}] {split} | {img_path.name} | Boxes: {len(boxes)}"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(
            vis,
            status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Display
        cv2.imshow(win_name, vis)
        
        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('n') or key == 83:  # 'n' or right arrow
            idx = (idx + 1) % len(images)
        elif key == ord('p') or key == 81:  # 'p' or left arrow
            idx = (idx - 1) % len(images)
    
    cv2.destroyAllWindows()
    print("\n[DONE] Visualization closed")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO dataset with bounding boxes"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to YOLO dataset folder (containing images/, labels/, data.yaml)"
    )
    
    args = parser.parse_args()
    visualize_dataset(args.dataset)


if __name__ == "__main__":
    main()

