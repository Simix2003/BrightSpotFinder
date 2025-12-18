from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import shutil
import time

MODEL_PATH = "Models/yolov8n_residual/weights/best.pt"
#IMAGE_DIR = "D:/Imix/Lavori/Capgemini/3SUN/Immagini/IMAGE/TEST"
#OUT_DIR = "D:/Imix/Lavori/Capgemini/3SUN/Immagini/IMAGE/OUTPUT"
IMAGE_DIR = "E:/foto_sofia"
OUT_DIR = "E:/foto_sofia_output"

# Create output directories
images_dir = os.path.join(OUT_DIR, "Image")
noted_dir = os.path.join(OUT_DIR, "Noted")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(noted_dir, exist_ok=True)

model = YOLO(MODEL_PATH)

# Statistics
total_images = 0
images_with_bright_spot = 0
copy_errors = 0
save_errors = 0

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    total_images += 1
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    # Run inference (use slightly lower threshold to catch edge cases, then filter to > 0.5)
    results = model(img_path, conf=0.3, imgsz=1024)
    
    # Check if any detection has confidence > 50%
    has_bright_spot = False
    high_conf_boxes = []
    
    # Get image dimensions
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Failed to load image: {img_path}")
        continue
    
    img_height, img_width = img.shape[:2]
    
    # Process detections and filter by confidence > 50%
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > 0.5:  # Confidence threshold (> 50%)
                has_bright_spot = True
                
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                high_conf_boxes.append((x1, y1, x2, y2, class_id, conf))
    
    # If bright spot detected, copy image and save annotated version
    if has_bright_spot:
        images_with_bright_spot += 1
        
        # Copy original image to Image folder with retry logic
        dst_img_path = os.path.join(images_dir, img_name)
        copy_success = False
        max_retries = 3
        retry_delay = 0.5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Check if destination file exists and is locked
                if os.path.exists(dst_img_path):
                    # Try to remove it first if it exists
                    try:
                        os.remove(dst_img_path)
                    except PermissionError:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            raise
                
                shutil.copy2(img_path, dst_img_path)
                copy_success = True
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"[WARN] Permission denied copying {img_name}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"[ERROR] Failed to copy {img_name} after {max_retries} attempts: {e}")
                    copy_errors += 1
            except Exception as e:
                print(f"[ERROR] Unexpected error copying {img_name}: {e}")
                copy_errors += 1
                break
        
        # Create annotated image with bounding boxes drawn
        annotated_img = img.copy()
        
        for x1, y1, x2, y2, class_id, conf in high_conf_boxes:
            # Draw bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"bright_spot {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(
                annotated_img,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_img,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        # Save annotated image to Noted folder with error handling
        noted_img_path = os.path.join(noted_dir, img_name)
        save_success = False
        
        for attempt in range(max_retries):
            try:
                # Check if destination file exists and is locked
                if os.path.exists(noted_img_path):
                    try:
                        os.remove(noted_img_path)
                    except PermissionError:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            raise
                
                if cv2.imwrite(noted_img_path, annotated_img):
                    save_success = True
                    break
                else:
                    raise Exception("cv2.imwrite returned False")
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"[WARN] Permission denied saving {img_name}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"[ERROR] Failed to save annotated {img_name} after {max_retries} attempts: {e}")
                    save_errors += 1
            except Exception as e:
                print(f"[ERROR] Unexpected error saving annotated {img_name}: {e}")
                save_errors += 1
                break
        
        if copy_success and save_success:
            print(f"[✓] {img_name} - {len(high_conf_boxes)} bright spot(s) detected")
        else:
            status_parts = []
            if not copy_success:
                status_parts.append("copy failed")
            if not save_success:
                status_parts.append("save failed")
            print(f"[⚠] {img_name} - {len(high_conf_boxes)} bright spot(s) detected, but {' and '.join(status_parts)}")
    else:
        print(f"[✗] {img_name} - No bright spot detected (confidence > 50%)")

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total images scanned: {total_images}")
print(f"Images with bright spot detected: {images_with_bright_spot}")
print(f"Images without bright spot: {total_images - images_with_bright_spot}")
print(f"Success rate: {(images_with_bright_spot / total_images * 100):.1f}%" if total_images > 0 else "N/A")
if copy_errors > 0 or save_errors > 0:
    print(f"\nErrors encountered:")
    if copy_errors > 0:
        print(f"  Copy errors: {copy_errors}")
    if save_errors > 0:
        print(f"  Save errors: {save_errors}")
print("="*60)
print(f"\nOutput saved to:")
print(f"  Original images: {images_dir}")
print(f"  Annotated images (with bounding boxes): {noted_dir}")
