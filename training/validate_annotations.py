import cv2
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_yolo_annotation(annotation_path, img_width, img_height):
    """Load YOLO format annotation and convert to pixel coordinates"""
    boxes = []
    if annotation_path.exists():
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1]) * img_width
                    center_y = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # Convert to corner coordinates
                    x1 = int(center_x - width / 2)
                    y1 = int(center_y - height / 2)
                    x2 = int(center_x + width / 2)
                    y2 = int(center_y + height / 2)
                    
                    boxes.append((class_id, x1, y1, x2, y2))
    return boxes

def count_classes(boxes):
    """Count occurrences of each class ID"""
    class_counts = defaultdict(int)
    for class_id, _, _, _, _ in boxes:
        class_counts[class_id] += 1
    return class_counts

def main():
    image_dir = Path("training/data/transferlearning/Test2/stg1/good")
    label_dir = Path("training/data/transferlearning/Test2/stg1/labels")
    
    # Class mapping:
    # Class 0 is "20", 1 is "3", 2 is "11", 3 is "6", 4 is "dart", 5 is "9" and 6 is "15"
    class_names = {
        0: "20",
        1: "3",
        2: "11", 
        3: "6",
        4: "dart",
        5: "9",
        6: "15"
    }
    DART_CLASS_ID = 4
    
    # Toggle states for display
    show_boxes = True
    show_circles = True
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    print(f"Found {len(image_files)} images to validate")
    print("Controls:")
    print("  'n' - next image")
    print("  'p' - previous image") 
    print("  'b' - toggle bounding boxes")
    print("  'c' - toggle center circles")
    print("  'o' - move to okay folder")
    print("  'd' - move to bad folder")
    print("  'q' - quit")
    
    # Create a colormap for different classes
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8).tolist()
    
    i = 0
    while i < len(image_files):
        image_path = image_files[i]
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            i += 1
            continue
            
        img_height, img_width = image.shape[:2]
        
        # Load annotations
        annotation_path = label_dir / f"{image_path.stem}.txt"
        boxes = load_yolo_annotation(annotation_path, img_width, img_height)
        
        # Count classes
        class_counts = count_classes(boxes)
        dart_count = sum(1 for class_id, _, _, _, _ in boxes if class_id == DART_CLASS_ID)
        
        # Draw bounding boxes and circles
        display_image = image.copy()
        for class_id, x1, y1, x2, y2 in boxes:
            color = colors[class_id]
            
            # Draw bounding box if enabled
            if show_boxes:
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                
                # Get proper class name from mapping
                label = class_names.get(class_id, f"Unknown {class_id}")
                cv2.putText(display_image, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center circle if enabled
            if show_circles:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(display_image, (center_x, center_y), 5, color, -1)  # Filled circle                cv2.circle(display_image, (center_x, center_y), 5, (255, 255, 255), 1)  # White border
        
        # Add filename and stats to image
        cv2.putText(display_image, f"{image_path.name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(display_image, f"Total objects: {len(boxes)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(display_image, f"Darts: {dart_count}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display toggle states
        box_status = "ON" if show_boxes else "OFF"
        circle_status = "ON" if show_circles else "OFF"
        cv2.putText(display_image, f"Boxes: {box_status} | Circles: {circle_status}", (10, 110),                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Validation", display_image)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            i += 1
        elif key == ord('b'):
            show_boxes = not show_boxes
            continue  # Redraw current image
        elif key == ord('c'):
            show_circles = not show_circles
            continue  # Redraw current image
        elif key == ord('o'):
            # Move the current image to "okay" folder
            okay_folder = Path("training/data/transferlearning/stg3/okay")
            new_path = okay_folder / image_path.name
            image_path.rename(new_path)
            i += 1
        elif key == ord('d'):
            # Move the current image to "bad" folder
            bad_folder = Path("training/data/transferlearning/stg3/bad")
            new_path = bad_folder / image_path.name
            image_path.rename(new_path)
            i += 1
        elif key == ord('p') and i > 0:
            i -= 1  # Go back to previous image
    
    cv2.destroyAllWindows()
    print("Validation complete!")

if __name__ == "__main__":
    main()
