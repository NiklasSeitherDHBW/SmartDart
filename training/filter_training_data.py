"""
Clean version of training data filter script with annotation visualization.
Allows manual review and filtering of training images and labels.
"""

import cv2
import os
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional
import sys

# Configuration constants
CONFIG = {
    "WINDOW_NAME": "Image Viewer",
    "ERROR_IMG_SIZE": (800, 800),
    "FONT": cv2.FONT_HERSHEY_SIMPLEX,
    "FONT_SCALE": 0.5,
    "FONT_THICKNESS": 2,      
    "COLORS": {
        "GREEN": (0, 255, 0),
        "BRIGHT_GREEN": (0, 255, 128),  # Replaced RED with more readable bright green
        "BRIGHT_BLUE": (255, 100, 0),
        "YELLOW": (0, 255, 255),
        "MAGENTA": (255, 0, 255),
        "CYAN": (255, 255, 0),
        "ORANGE": (0, 165, 255),
        "WHITE": (255, 255, 255),
        "LIME": (0, 255, 128),
        "PINK": (255, 20, 147),
    },
    "TEXT_POSITIONS": {
        "FILENAME": (10, 30),
        "ERROR": (10, 60),
        "ANNOTATIONS": (10, 90),
    },
    "BBOX_THICKNESS": 2,
    "LABEL_OFFSET": (5, -5),
    "CLASS_MAPPING": {
        0: "20",
        1: "3",
        2: "11",
        3: "6",
        4: "dart",
        5: "9",
        6: "15"
    }
}

# Key mappings
KEYS = {
    "QUIT": ord("q"),
    "DELETE": ord("d"),
    "KEEP": ord("c"),
    "OKAY": ord("o"),
    "TOGGLE_ANNOTATIONS": ord("a"),
}


class TrainingDataFilter:
    """Handles filtering and review of training data images and labels."""
    
    def __init__(self, data_root: Path, start_index: int = 0, show_annotations: bool = True):
        self.data_root = Path(data_root)
        self.imgs_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"
        self.perfect_imgs_file = self.data_root / "perfect_images.txt"
        self.start_index = start_index
        self.perfect_imgs = self._load_perfect_images()
        self.show_annotations = show_annotations
        
    def _load_perfect_images(self) -> List[str]:
        """Load existing perfect images list if it exists."""
        if self.perfect_imgs_file.exists():
            with open(self.perfect_imgs_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        return []
    
    def _save_perfect_images(self) -> None:
        """Save the current perfect images list to file."""
        return None  # TODO: Improve saving logic
        with open(self.perfect_imgs_file, 'w') as f:
            for img in self.perfect_imgs:
                f.write(f"{img}\n")
    
    def _get_file_pairs(self) -> Tuple[List[Path], List[Path]]:
        """Get matching image and label file pairs."""
        if not self.imgs_dir.exists() or not self.labels_dir.exists():
            raise FileNotFoundError(f"Images or labels directory not found in {self.data_root}")
            
        images = sorted([f for f in self.imgs_dir.iterdir() if f.suffix.lower() == '.jpg'])
        labels = sorted([f for f in self.labels_dir.iterdir() if f.suffix.lower() == '.txt'])
        
        # Apply start index
        images = images[self.start_index:]
        labels = labels[self.start_index:]
        
        return images, labels
    
    def _validate_file_pairs(self, images: List[Path], labels: List[Path]) -> None:
        """Validate that image and label files match properly."""
        if len(images) != len(labels):
            raise ValueError(f"Mismatch: {len(images)} images vs {len(labels)} labels")
        
        mismatches = []
        for img_path, label_path in zip(images, labels):
            img_stem = img_path.stem
            label_stem = label_path.stem
            if img_stem != label_stem:
                mismatches.append((img_path, label_path))
        
        if mismatches:
            print("File mismatches found:")
            for img, label in mismatches:
                print(f"  {img.name} ↔ {label.name}")
            raise ValueError("Image and label files don't match")
    
    def _load_annotations(self, label_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """Load YOLO format annotations from label file.
        
        Returns:
            List of tuples: (class_id, x_center, y_center, width, height) - all normalized 0-1
        """
        annotations = []
        if not label_path.exists():
            return annotations
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            annotations.append((class_id, x_center, y_center, width, height))
        except (ValueError, IOError) as e:
            print(f"Error reading annotations from {label_path}: {e}")
        
        return annotations
    
    def _draw_annotations(self, img: np.ndarray, annotations: List[Tuple[int, float, float, float, float]]) -> np.ndarray:
        """Draw bounding box annotations on the image.
        
        Args:
            img: Input image
            annotations: List of (class_id, x_center, y_center, width, height) normalized 0-1
            
        Returns:
            Image with annotations drawn
        """
        if not annotations:
            return img
        
        img_copy = img.copy()
        img_height, img_width = img.shape[:2]          # High-contrast color palette optimized for dark backgrounds
        colors = [
            CONFIG["COLORS"]["WHITE"],          # Class 0 - White
            CONFIG["COLORS"]["YELLOW"],         # Class 1 - Bright Yellow  
            CONFIG["COLORS"]["LIME"],           # Class 2 - Bright Lime Green
            CONFIG["COLORS"]["CYAN"],           # Class 3 - Bright Cyan
            CONFIG["COLORS"]["MAGENTA"],        # Class 4 - Bright Magenta
            CONFIG["COLORS"]["ORANGE"],         # Class 5 - Bright Orange
            CONFIG["COLORS"]["WHITE"],          # Class 6 - White
            CONFIG["COLORS"]["PINK"],           # Class 7 - Hot Pink
        ]
        
        for class_id, x_center, y_center, width, height in annotations:
            # Convert normalized coordinates to pixel coordinates
            x_center_px = int(x_center * img_width)
            y_center_px = int(y_center * img_height)
            width_px = int(width * img_width)
            height_px = int(height * img_height)
            
            # Calculate bounding box corners
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            
            # Choose color based on class_id
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, CONFIG["BBOX_THICKNESS"])
            
            # Draw class label
            label = CONFIG["CLASS_MAPPING"][class_id]
            label_x = x1 + CONFIG["LABEL_OFFSET"][0]
            label_y = y1 + CONFIG["LABEL_OFFSET"][1]
            
            # Ensure label is within image bounds
            if label_y < 20:
                label_y = y2 + 20
            
            cv2.putText(
                img_copy, label,
                (label_x, label_y),
                CONFIG["FONT"], CONFIG["FONT_SCALE"],
                color, CONFIG["FONT_THICKNESS"],
                cv2.LINE_AA
            )
        
        return img_copy
    
    def _load_image_safely(self, img_path: Path) -> Tuple[Optional[np.ndarray], str]:
        """Load an image safely, returning error message if failed."""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None, f"Could not read image: {img_path.name}"
            return img, ""
        except Exception as e:
            return None, f"Error loading {img_path.name}: {str(e)}"
    
    def _create_error_image(self, error_msg: str, filename: str) -> np.ndarray:
        """Create a placeholder image to display error information."""
        img = np.zeros((*CONFIG["ERROR_IMG_SIZE"], 3), dtype=np.uint8)
        
        # Add filename
        cv2.putText(
            img, f"Image: {filename}",
            CONFIG["TEXT_POSITIONS"]["FILENAME"],
            CONFIG["FONT"], CONFIG["FONT_SCALE"],
            CONFIG["COLORS"]["GREEN"], CONFIG["FONT_THICKNESS"],
            cv2.LINE_AA
        )
          # Add error message
        cv2.putText(
            img, error_msg,
            CONFIG["TEXT_POSITIONS"]["ERROR"],
            CONFIG["FONT"], CONFIG["FONT_SCALE"],
            CONFIG["COLORS"]["ORANGE"], CONFIG["FONT_THICKNESS"],
            cv2.LINE_AA
        )
        
        return img
    
    def _add_filename_overlay(self, img: np.ndarray, filename: str, annotation_count: int = 0, show_annotations: bool = True) -> np.ndarray:
        """Add filename and annotation info overlay to image."""
        img_copy = img.copy()
        cv2.putText(
            img_copy, f"Image: {filename}",
            CONFIG["TEXT_POSITIONS"]["FILENAME"],
            CONFIG["FONT"], CONFIG["FONT_SCALE"],
            CONFIG["COLORS"]["GREEN"], CONFIG["FONT_THICKNESS"],
            cv2.LINE_AA
        )
          # Show annotation info
        annotation_text = f"Annotations: {annotation_count} ({'ON' if show_annotations else 'OFF'})"
        cv2.putText(
            img_copy, annotation_text,
            CONFIG["TEXT_POSITIONS"]["ANNOTATIONS"],
            CONFIG["FONT"], CONFIG["FONT_SCALE"],
            CONFIG["COLORS"]["CYAN"], CONFIG["FONT_THICKNESS"],
            cv2.LINE_AA
        )
        
        return img_copy
    
    def _handle_user_input(self, key: int, img_path: Path, label_path: Path) -> bool:
        """Handle user keyboard input. Returns True to continue, False to quit."""
        if key == KEYS["QUIT"]:
            print("Exiting viewer.")
            cv2.destroyAllWindows()
            return False
            
        elif key == KEYS["DELETE"]:
            print(f"Deleting {img_path.name} and {label_path.name}")
            try:
                os.remove(img_path)
                os.remove(label_path)
                # Remove from perfect images if it was there
                if img_path.name in self.perfect_imgs:
                    self.perfect_imgs.remove(img_path.name)
                    self._save_perfect_images()
            except OSError as e:
                print(f"Error deleting files: {e}")
                
        elif key == KEYS["KEEP"]:
            print(f"Keeping {img_path.name} and {label_path.name}")
            if img_path.name not in self.perfect_imgs:
                self.perfect_imgs.append(img_path.name)
                self._save_perfect_images()

        elif key == KEYS["OKAY"]:
            print(f"Moving image {img_path.name} to okay images")
            os.rename(img_path, self.data_root / "okay_images" / img_path.name)
            os.rename(label_path, self.data_root / "okay_labels" / label_path.name)
                
        elif key == KEYS["TOGGLE_ANNOTATIONS"]:
            self.show_annotations = not self.show_annotations
            print(f"Annotations {'ON' if self.show_annotations else 'OFF'}")
            return "redraw"  # Special return value to trigger redraw
        
        return True
    
    def run(self) -> None:
        """Main execution loop for filtering training data."""
        try:
            images, labels = self._get_file_pairs()
            self._validate_file_pairs(images, labels)
            
            print(f"Found {len(images)} image-label pairs to review")
            print("Controls:")
            print("  'q' - Quit")
            print("  'd' - Delete current pair")
            print("  'c' - Keep current pair (mark as perfect)")
            print("  'o' - Move current pair to okay images")
            print("  'a' - Toggle annotation display")
            print("  Any other key - Skip to next")
            
            i = 0
            while i < len(images):
                img_path = images[i]
                label_path = labels[i]
                
                img, error_msg = self._load_image_safely(img_path)
                annotations = self._load_annotations(label_path)
                
                if img is None:
                    print(f"Error: {error_msg}")
                    display_img = self._create_error_image(error_msg, img_path.name)
                else:
                    # Draw annotations if enabled
                    if self.show_annotations:
                        img_with_annotations = self._draw_annotations(img, annotations)
                    else:
                        img_with_annotations = img
                    
                    display_img = self._add_filename_overlay(
                        img_with_annotations, img_path.name, len(annotations), self.show_annotations
                    )
                
                cv2.imshow(CONFIG["WINDOW_NAME"], display_img)
                key = cv2.waitKey(0) & 0xFF
                
                result = self._handle_user_input(key, img_path, label_path)
                if result is False:  # Quit
                    break
                elif result == "redraw":  # Redraw current image
                    continue
                else:  # Move to next image
                    i += 1
                    
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cv2.destroyAllWindows()


def main():
    """Main entry point."""
    # Configuration
    start_index = 0  # You can modify this to resume from a specific index
    data_root = Path("training/data/transferlearning/Test2/stg1")

    Path(data_root / "okay_images").mkdir(parents=True, exist_ok=True)
    Path(data_root / "okay_labels").mkdir(parents=True, exist_ok=True)
    
    # Add parent directory to path for utils import (if needed)
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Create and run the filter
    filter_tool = TrainingDataFilter(data_root, start_index)
    filter_tool.run()


if __name__ == "__main__":
    main()
