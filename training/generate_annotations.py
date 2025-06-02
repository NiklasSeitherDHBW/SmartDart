import cv2
import os
from pathlib import Path
from utils import predict
import numpy as np
from collections import defaultdict

def create_yolo_annotation(image_path, results, output_dir):
    """
    Create YOLO format annotation file from YOLO results
    Filter to keep only:
    - Highest confidence detection for each number
    - Top 3 highest confidence dart detections
    """
    image = cv2.imread(str(image_path))
    img_height, img_width = image.shape[:2]
    
    # Create annotation file path
    annotation_path = output_dir / f"{image_path.stem}.txt"
    
    # Group detections by class and sort by confidence
    numbers_by_class = defaultdict(list)
    darts = []
    
    # Class mapping:
    # Class 0 is "20", 1 is "3", 2 is "11", 3 is "6", 4 is "dart", 5 is "9" and 6 is "15"
    DART_CLASS_ID = 4
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Get box information
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Calculate YOLO format coordinates
                center_x = (x1 + x2) / 2 / img_width
                center_y = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Detection object
                detection = {
                    'class_id': class_id,
                    'confidence': confidence,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height
                }
                
                # Check if this is a dart based on the correct class ID
                if class_id == DART_CLASS_ID:
                    darts.append(detection)
                else:
                    # It's a number on the dartboard
                    numbers_by_class[class_id].append(detection)
    
    # Filter detections
    filtered_detections = []
    
    # 1. For numbers, keep only the highest confidence for each unique class
    for class_id, detections in numbers_by_class.items():
        if detections:
            # Sort by confidence (highest first) and keep only the first one
            best_detection = sorted(detections, key=lambda x: x['confidence'], reverse=True)[0]
            filtered_detections.append(best_detection)
    
    # 2. For darts, keep only top 3 highest confidence
    if darts:
        top_darts = sorted(darts, key=lambda x: x['confidence'], reverse=True)[:3]
        filtered_detections.extend(top_darts)
    
    # Write filtered detections to file
    with open(annotation_path, 'w') as f:
        for detection in filtered_detections:
            f.write(f"{detection['class_id']} {detection['center_x']:.6f} "
                   f"{detection['center_y']:.6f} {detection['width']:.6f} "
                   f"{detection['height']:.6f}\n")
    
    return len(filtered_detections)

def main():
    # Paths
    input_dir = Path("training/data/transferlearning/good")
    output_dir = Path("training/data/transferlearning/labels")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize predictor
    predictor = predict.Predictor(model_path="models/yolo8n.pt")

    # Get all image files
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    print(f"Found {len(image_files)} images to annotate")

    for i, image_path in enumerate(image_files):
        if "3102" in image_path.name:
            print("Switching model to finetuned version")
            predictor = predict.Predictor(model_path="models/yolo8n-finetune.pt")
        
        print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load {image_path}")
            continue
        
        # Predict
        results = predictor.predict(image)
        
        # Create annotation
        num_annotations = create_yolo_annotation(image_path, results, output_dir)
        
        print(f"Created annotation for {image_path.name} with {num_annotations} objects")
    
    print(f"Annotation generation complete! Labels saved to {output_dir}")

if __name__ == "__main__":
    main()
