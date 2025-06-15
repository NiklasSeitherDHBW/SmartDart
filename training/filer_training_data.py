import os
import sys
import cv2
import shutil
import glob
from pathlib import Path
import numpy as np

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predict import Predictor
from utils.calibration import CameraCalibration

class AnnotationSorter:
    def __init__(self, model_path, raw_images_dir, base_output_dir):
        """
        Initialize the annotation sorter.
        
        Args:
            model_path: Path to the YOLO model
            raw_images_dir: Directory containing raw images to process
            base_output_dir: Base directory where sorted images will be saved
        """
        self.model_path = model_path
        self.raw_images_dir = Path(raw_images_dir)
        self.base_output_dir = Path(base_output_dir)
        
        # Create output directories if they don't exist
        self.good_dir = self.base_output_dir / "good"
        self.okay_dir = self.base_output_dir / "okay"
        self.bad_dir = self.base_output_dir / "bad"
        
        for dir_path in [self.good_dir, self.okay_dir, self.bad_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the predictor
        print(f"Loading YOLO model from: {model_path}")
        self.predictor = Predictor(model_path)
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(glob.glob(str(self.raw_images_dir / ext)))
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images to process")
        
        self.current_index = 0
        
        self.calib = CameraCalibration("resources/dartboard-gerade.jpg")
        
        # Toggle states for visualization
        self.show_boxes = True
        self.show_circles = True
        
    def draw_predictions(self, image, results):
        """
        Use Ultralytics' built-in visualization to draw YOLO predictions on the image,
        then add center dots to each bounding box. Both can be toggled on/off.
        
        Args:
            image: Original image
            results: YOLO prediction results
            
        Returns:
            Image with annotations drawn based on toggle states
        """
        # Get the first result (assuming single image prediction)
        if len(results) > 0:
            result = results[0]
            
            if self.show_boxes:
                # Use Ultralytics' built-in plot method for better visualization
                annotated_image = result.plot(
                    conf=False,          # Show confidence scores
                    labels=True,        # Show class labels
                    boxes=True,         # Show bounding boxes
                    line_width=3,       # Line thickness
                )
            else:
                # If boxes are disabled, start with original image
                annotated_image = image.copy()
            
            # Add center dots to each bounding box if enabled
            if self.show_circles and result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    
                    # Calculate center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Draw center dot (filled circle)
                    cv2.circle(annotated_image, (center_x, center_y), 5, (255, 0, 0), -1)  # Blue dot, filled
                    # Draw small white outline for better visibility
                    cv2.circle(annotated_image, (center_x, center_y), 5, (255, 255, 255), 1)  # White outline
            
            return annotated_image
        else:
            # Return original image if no detections
            return image.copy()
    
    def copy_image_to_folder(self, image_path, destination_folder):
        """
        Copy image to the specified folder.
        
        Args:
            image_path: Source image path
            destination_folder: Destination folder path
        """
        filename = os.path.basename(image_path)
        destination_path = destination_folder / filename
        
        try:
            shutil.copy2(image_path, destination_path)
            print(f"Copied {filename} to {destination_folder.name}")
            return True
        except Exception as e:
            print(f"Error copying {filename}: {e}")
            return False
    
    def process_images(self):
        """
        Main loop to process images and handle user input.
        """
        if not self.image_files:
            print("No images found in the specified directory!")
            return
        
        print("\nControls:")
        print("  'g' - Copy to 'good' folder")
        print("  'o' - Copy to 'okay' folder") 
        print("  'b' - Copy to 'bad' folder")
        print("  'n' - Skip to next image (no copy)")
        print("  'p' - Go to previous image")
        print("  'x' - Toggle bounding boxes")
        print("  'c' - Toggle center circles")
        print("  'q' - Quit")
        print("  Space - Skip to next image (no copy)")
        print("\nPress any key to start...")
        
        cv2.namedWindow('Annotation Sorter', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Annotation Sorter', 800, 800)
        
        while self.current_index < len(self.image_files):
            image_path = self.image_files[self.current_index]
            filename = os.path.basename(image_path)
            
            print(f"\nProcessing image {self.current_index + 1}/{len(self.image_files)}: {filename}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                self.current_index += 1
                continue
            
            success, image = self.calib.initial_calibration(image)
            if not success:
                # Display black image if calibration fails and tell user that it failed
                print(f"Calibration failed for image: {image_path}. Skipping...")
                image = np.zeros_like(image)
                cv2.putText(image, "Calibration failed", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                results = []  # No predictions for failed calibration
            else:
                # Get predictions
                print("Generating predictions...")
                results = self.predictor.predict(image)
            
            # Inner loop for handling display and toggles without re-prediction
            while True:
                # Draw annotations
                annotated_image = self.draw_predictions(image, results)
                
                # Add info text on image
                info_text = f"Image {self.current_index + 1}/{len(self.image_files)}: {filename}"
                cv2.putText(annotated_image, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Add toggle status
                toggle_text = f"Boxes: {'ON' if self.show_boxes else 'OFF'} | Circles: {'ON' if self.show_circles else 'OFF'}"
                cv2.putText(annotated_image, toggle_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                control_text = "g=good, o=okay, b=bad, x=boxes, c=circles, n/space=skip, p=prev, q=quit"
                cv2.putText(annotated_image, control_text, (10, annotated_image.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Display image
                cv2.imshow('Annotation Sorter', annotated_image)
                
                # Wait for key press
                key = cv2.waitKey(0) & 0xFF
                
                # Handle toggle keys (these don't break the inner loop)
                if key == ord('x'):
                    # Toggle bounding boxes
                    self.show_boxes = not self.show_boxes
                    print(f"Bounding boxes: {'ON' if self.show_boxes else 'OFF'}")
                    continue  # Stay in inner loop, just refresh display
                    
                elif key == ord('c'):
                    # Toggle center circles
                    self.show_circles = not self.show_circles
                    print(f"Center circles: {'ON' if self.show_circles else 'OFF'}")
                    continue  # Stay in inner loop, just refresh display
                  # All other keys break out of inner loop
                break
            
            # Handle keys that affect image navigation/sorting
            if key == ord('g'):
                # Copy to good folder
                if self.copy_image_to_folder(image_path, self.good_dir):
                    self.current_index += 1
                    
            elif key == ord('o'):
                # Copy to okay folder
                if self.copy_image_to_folder(image_path, self.okay_dir):
                    self.current_index += 1
                    
            elif key == ord('b'):
                # Copy to bad folder
                if self.copy_image_to_folder(image_path, self.bad_dir):
                    self.current_index += 1
                    
            elif key == ord('n') or key == ord(' '):
                # Skip to next
                print(f"Skipped {filename}")
                self.current_index += 1
                
            elif key == ord('p'):
                # Go to previous
                if self.current_index > 0:
                    self.current_index -= 1
                    print("Going to previous image...")
                else:
                    print("Already at first image!")
                    
            elif key == ord('q') or key == 27:  # 'q' or Escape
                print("Quitting...")
                break
            
            else:
                print(f"Unknown key pressed. Use g/o/b to sort, x=toggle boxes, c=toggle circles, n/space to skip, p for previous, q to quit.")
        
        cv2.destroyAllWindows()
        print(f"\nFinished processing. Images sorted into:")
        print(f"  Good: {self.good_dir}")
        print(f"  Okay: {self.okay_dir}")
        print(f"  Bad: {self.bad_dir}")

def main():
    # Configuration
    MODEL_PATH = Path("models/stg4.pt")
    RAW_IMAGES_DIR = Path("training/data/transferlearning/Test2/stg1/raw")
    BASE_OUTPUT_DIR = Path("training/data/transferlearning/Test2/stg1")

    Path(BASE_OUTPUT_DIR / "good").mkdir(parents=True, exist_ok=True)
    Path(BASE_OUTPUT_DIR / "okay").mkdir(parents=True, exist_ok=True)
    Path(BASE_OUTPUT_DIR / "bad").mkdir(parents=True, exist_ok=True)
    Path(BASE_OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    # Check if images directory exists
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"Error: Images directory not found at {RAW_IMAGES_DIR}")
        return
    
    # Create and run the sorter
    sorter = AnnotationSorter(MODEL_PATH, RAW_IMAGES_DIR, BASE_OUTPUT_DIR)
    sorter.process_images()

if __name__ == "__main__":
    main()
