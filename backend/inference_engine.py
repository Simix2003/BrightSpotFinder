from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import shutil
import time
from typing import List, Tuple, Optional
from models import ImageResult, DetectionBox


class InferenceEngine:
    def __init__(self, model_path: str):
        """Initialize the inference engine with a YOLO model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = YOLO(model_path)
        self.model_path = model_path
    
    def process_image(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        inference_conf: float = 0.3,
        imgsz: int = 1024
    ) -> ImageResult:
        """
        Process a single image and return detection results
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for detections (default 0.5)
            inference_conf: Confidence threshold for YOLO inference (default 0.3)
            imgsz: Image size for inference (default 1024)
        
        Returns:
            ImageResult with detection information
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Run inference
        results = self.model(image_path, conf=inference_conf, imgsz=imgsz)
        
        # Load image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        detections = []
        has_bright_spot = False
        
        # Process detections and filter by confidence threshold
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf > confidence_threshold:
                    has_bright_spot = True
                    
                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0])
                    
                    detections.append(DetectionBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=conf,
                        class_id=class_id
                    ))
        
        image_name = os.path.basename(image_path)
        return ImageResult(
            image_name=image_name,
            has_bright_spot=has_bright_spot,
            detections=detections,
            processed=True
        )
    
    def save_annotated_image(
        self,
        image_path: str,
        image_result: ImageResult,
        output_dir: str
    ) -> Tuple[str, str]:
        """
        Save original and annotated images to output directory
        
        Args:
            image_path: Path to source image
            image_result: ImageResult with detection data
            output_dir: Base output directory
        
        Returns:
            Tuple of (original_image_path, annotated_image_path)
        """
        # Create output directories
        images_dir = os.path.join(output_dir, "Image")
        noted_dir = os.path.join(output_dir, "Noted")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(noted_dir, exist_ok=True)
        
        image_name = image_result.image_name
        
        # Copy original image with retry logic
        dst_img_path = os.path.join(images_dir, image_name)
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
                
                shutil.copy2(image_path, dst_img_path)
                copy_success = True
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"[WARN] Permission denied copying {image_name}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"[ERROR] Failed to copy {image_name} after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                print(f"[ERROR] Unexpected error copying {image_name}: {e}")
                raise
        
        if not copy_success:
            raise RuntimeError(f"Failed to copy image {image_name} after {max_retries} attempts")
        
        # Create annotated image if bright spot detected
        if image_result.has_bright_spot:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image for annotation: {image_path}")
            
            annotated_img = img.copy()
            
            for detection in image_result.detections:
                # Draw bounding box
                x1, y1, x2, y2 = int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with confidence
                label = f"bright_spot {detection.confidence:.2f}"
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
            
            # Save annotated image with retry logic
            noted_img_path = os.path.join(noted_dir, image_name)
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
                        print(f"[WARN] Permission denied saving {image_name}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        print(f"[ERROR] Failed to save annotated {image_name} after {max_retries} attempts: {e}")
                        raise
                except Exception as e:
                    print(f"[ERROR] Unexpected error saving annotated {image_name}: {e}")
                    raise
            
            if not save_success:
                raise RuntimeError(f"Failed to save annotated image {image_name} after {max_retries} attempts")
            
            return (dst_img_path, noted_img_path)
        
        return (dst_img_path, None)
    
    def get_image_list(self, input_dir: str) -> List[str]:
        """
        Get list of image files from input directory
        
        Args:
            input_dir: Directory containing images
        
        Returns:
            List of image file paths
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        image_extensions = (".png", ".jpg", ".jpeg")
        image_files = []
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(input_dir, filename)
                image_files.append(image_path)
        
        return sorted(image_files)

