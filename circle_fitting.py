import numpy as np
import cv2
import os
import glob
from typing import List, Tuple, Optional

def fit_circle_through_dartboard_points(points: List[Tuple[int, int]], 
                                      center_estimate: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[int, int], float]:
    """
    Fits a circle through dartboard corner points (20, 6, 15, 3, 11, 9).
    Handles small angular deviations from perfect alignment.
    
    Args:
        points: List of (x, y) coordinates of the corner points
        center_estimate: Optional estimate of dartboard center
    
    Returns:
        ((center_x, center_y), radius) of the fitted circle
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a circle")
    
    points_array = np.array(points, dtype=np.float64)
    
    # Method 1: Algebraic circle fitting (works well with noise)
    x = points_array[:, 0]
    y = points_array[:, 1]
    
    # Set up the system of equations: x² + y² + Dx + Ey + F = 0
    A = np.column_stack([x, y, np.ones(len(points))])
    b = -(x**2 + y**2)
    
    # Solve using least squares
    try:
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        D, E, F = coeffs
        
        center_x = -D / 2
        center_y = -E / 2
        radius = np.sqrt((D**2 + E**2) / 4 - F)
        
        return (int(center_x), int(center_y)), radius
    
    except np.linalg.LinAlgError:
        # Fallback: use center estimate and average distance
        if center_estimate is None:
            center_x = np.mean(x)
            center_y = np.mean(y)
        else:
            center_x, center_y = center_estimate
        
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        radius = np.mean(distances)
        
        return (int(center_x), int(center_y)), radius

def draw_fitted_circle(image: np.ndarray, 
                      points: List[Tuple[int, int]], 
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
    """
    Draws the fitted circle on the image along with the corner points.
    
    Args:
        image: Input image
        points: Corner points used for fitting
        color: Circle color (BGR)
        thickness: Circle line thickness
    
    Returns:
        Image with drawn circle and points
    """
    result = image.copy()
    
    if len(points) < 3:
        return result
    
    try:
        center, radius = fit_circle_through_dartboard_points(points)
        
        # Draw the fitted circle
        cv2.circle(result, center, int(radius), color, thickness)
        
        # Draw center point
        cv2.circle(result, center, 5, (255, 0, 0), -1)
        
        # Draw the corner points
        for i, point in enumerate(points):
            cv2.circle(result, point, 8, (0, 0, 255), -1)
            cv2.putText(result, str(i+1), (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add info text
        info_text = f"Center: {center}, Radius: {radius:.1f}"
        cv2.putText(result, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    except Exception as e:
        print(f"Error fitting circle: {e}")
    
    return result

def validate_dartboard_circle(points: List[Tuple[int, int]], 
                             tolerance_ratio: float = 0.1) -> bool:
    """
    Validates if the fitted circle is reasonable for a dartboard.
    
    Args:
        points: Corner points
        tolerance_ratio: Allowed deviation from average radius
    
    Returns:
        True if the circle seems valid
    """
    if len(points) < 3:
        return False
    
    try:
        center, radius = fit_circle_through_dartboard_points(points)
        
        # Check if all points are within tolerance of the fitted circle
        for point in points:
            distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
            deviation = abs(distance - radius) / radius
            
            if deviation > tolerance_ratio:
                return False
        
        return True
    
    except:
        return False

# Example usage function
def process_dartboard_image(image_path: str, corner_points: List[Tuple[int, int]]):
    """
    Example function showing how to use the circle fitting on a dartboard image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Fit and draw circle
    result = draw_fitted_circle(image, corner_points)
    
    # Validate the fit
    is_valid = validate_dartboard_circle(corner_points)
    status_text = "Valid dartboard circle" if is_valid else "Invalid circle fit"
    cv2.putText(result, status_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_valid else (0, 0, 255), 2)
    
    # Display result
    cv2.imshow('Fitted Circle', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result

def load_yolo_labels(label_path: str, image_width: int, image_height: int) -> List[Tuple[int, int]]:
    """
    Load YOLO format labels and convert to pixel coordinates.
    
    Args:
        label_path: Path to the .txt label file
        image_width: Width of the corresponding image
        image_height: Height of the corresponding image
    
    Returns:
        List of (x, y) pixel coordinates for the labeled points
    """
    points = []
    
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return points
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # YOLO format: class_id center_x center_y [width height]
                    class_id = int(parts[0])
                    center_x = float(parts[1]) * image_width
                    center_y = float(parts[2]) * image_height
                    points.append((int(center_x), int(center_y)))
        
        print(f"Loaded {len(points)} labeled points from {label_path}")
    except Exception as e:
        print(f"Error loading labels: {e}")
    
    return points

def get_sample_image_and_labels(data_folder: str = "training/data/empty_board") -> Tuple[Optional[str], Optional[str]]:
    """
    Get a random sample image and its corresponding label file.
    
    Args:
        data_folder: Base path to the training data folder
    
    Returns:
        (image_path, label_path) tuple or (None, None) if not found
    """
    good_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")
    
    # Find all image files in the good folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(good_folder, ext)))
    
    if not image_files:
        print(f"No image files found in {good_folder}")
        return None, None
    
    # Pick the first image (you can modify this to pick randomly)
    sample_image = image_files[0]
    
    # Find corresponding label file
    image_name = os.path.splitext(os.path.basename(sample_image))[0]
    label_file = os.path.join(labels_folder, f"{image_name}.txt")
    
    if not os.path.exists(label_file):
        print(f"No corresponding label file found for {sample_image}")
        return sample_image, None
    
    return sample_image, label_file

def process_sample_dartboard():
    """
    Process a sample dartboard image from the training data.
    """
    # Get sample image and labels
    image_path, label_path = get_sample_image_and_labels()
    
    if image_path is None:
        print("No sample image found")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Processing image: {os.path.basename(image_path)}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Load labels if available
    corner_points = []
    if label_path:
        corner_points = load_yolo_labels(label_path, image.shape[1], image.shape[0])
        print(f"Loaded {len(corner_points)} corner points: {corner_points}")
    
    # If we have enough points, fit a circle
    if len(corner_points) >= 3:
        result = draw_fitted_circle(image, corner_points)
        
        # Validate the fit
        is_valid = validate_dartboard_circle(corner_points)
        status_text = "Valid dartboard circle" if is_valid else "Invalid circle fit"
        cv2.putText(result, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_valid else (0, 0, 255), 2)
        
        # Add image filename to display
        filename_text = f"File: {os.path.basename(image_path)}"
        cv2.putText(result, filename_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display result
        cv2.imshow('Sample Dartboard with Fitted Circle', result)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return result
    else:
        print("Not enough corner points to fit a circle")
        # Just display the original image with any available points
        result = image.copy()
        for i, point in enumerate(corner_points):
            cv2.circle(result, point, 8, (0, 0, 255), -1)
            cv2.putText(result, str(i+1), (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Sample Dartboard', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return result

# Add this at the end for easy testing
if __name__ == "__main__":
    # Process a sample dartboard image
    process_sample_dartboard()
