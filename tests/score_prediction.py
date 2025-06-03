import numpy as np
import cv2
import os
import glob
from typing import List, Tuple, Optional
import math

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

def create_dartboard_template(radius: int = 300) -> np.ndarray:
    """
    Create a dartboard template with all field boundaries.
    
    Args:
        radius: Radius of the outer circle
    
    Returns:
        Template image with dartboard fields drawn
    """
    # Create blank image (square with some padding)
    size = radius * 2 + 100
    template = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    
    # Fine-tuned dartboard dimensions based on regulation measurements
    outer_double = radius                    # Outer edge (100%)
    inner_double = int(radius * 0.95)        # Inner edge of double (95%)
    
    # Refined triple ring position - moved inward
    outer_treble = int(radius * 0.625)       # Outer edge of treble (62.5%)
    inner_treble = int(radius * 0.56)        # Inner edge of treble (56%)
    
    # Much smaller bullseye area
    outer_bull = int(radius * 0.10)          # Outer bull/25 zone (10%) 
    inner_bull = int(radius * 0.04)          # Inner bull/bullseye (4%)
    
    # Draw concentric circles with corrected proportions
    cv2.circle(template, center, outer_double, (255, 255, 255), 2)    # Outer wire
    cv2.circle(template, center, inner_double, (255, 255, 255), 2)    # Double ring inner
    cv2.circle(template, center, outer_treble, (255, 255, 255), 2)    # Treble ring outer
    cv2.circle(template, center, inner_treble, (255, 255, 255), 2)    # Treble ring inner
    cv2.circle(template, center, outer_bull, (255, 255, 255), 2)      # Outer bull
    cv2.circle(template, center, inner_bull, (255, 255, 255), 2)      # Inner bull
    
    # Standard dartboard number sequence: 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5
    
    # Draw radial lines at segment BOUNDARIES (between numbers)
    # Start at -99 degrees (9 degrees before top) and increment by 18 degrees
    # This draws lines at segment boundaries rather than through their centers
    for i in range(20):
        angle = (i * 18 - 99) * math.pi / 180  # Start at boundary, go clockwise
        x1 = int(center[0] + inner_bull * math.cos(angle))
        y1 = int(center[1] + inner_bull * math.sin(angle))
        x2 = int(center[0] + outer_double * math.cos(angle))
        y2 = int(center[1] + outer_double * math.sin(angle))
        cv2.line(template, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
    return template

def calculate_dartboard_transform(reference_points: List[Tuple[int, int]], 
                                template_radius: int = 300) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """
    Calculate transformation matrix to fit template to detected dartboard.
    """
    # Fit circle through reference points
    detected_center, detected_radius = fit_circle_through_dartboard_points(reference_points)
    
    # Calculate scale factor
    scale = detected_radius / template_radius
    
    # Simplified rotation calculation
    rotation_angle = 0
    
    if len(reference_points) >= 1:
        # Use the first reference point to estimate rotation
        first_point = reference_points[0]
        dx = first_point[0] - detected_center[0]
        dy = first_point[1] - detected_center[1]
        
        # Calculate angle of first reference point
        ref_angle = math.atan2(dy, dx)
        
        # Assume first reference point should align with a dartboard segment boundary
        # Try to align it to the nearest 18-degree increment
        ref_angle_deg = math.degrees(ref_angle)
        if ref_angle_deg < 0:
            ref_angle_deg += 360
            
        # Find nearest segment boundary (every 18 degrees starting from -9)
        segment_angles = [(i * 18 - 9) % 360 for i in range(20)]
        nearest_segment = min(segment_angles, key=lambda x: min(abs(ref_angle_deg - x), abs(ref_angle_deg - x + 360), abs(ref_angle_deg - x - 360)))
        
        # Calculate rotation needed
        rotation_needed = nearest_segment - ref_angle_deg
        
        # Normalize to -180 to 180 range
        if rotation_needed > 180:
            rotation_needed -= 360
        elif rotation_needed < -180:
            rotation_needed += 360
            
        rotation_angle = math.radians(rotation_needed)
        
        print(f"First reference point: {ref_angle_deg:.1f}°")
        print(f"Nearest segment: {nearest_segment:.1f}°")
        print(f"Rotation needed: {rotation_needed:.1f}°")
    
    # Create transformation matrix
    template_center = (template_radius + 50, template_radius + 50)
    
    # Combine scale, rotation, and translation
    M_combined = cv2.getRotationMatrix2D(template_center, math.degrees(rotation_angle), scale)
    M_combined[0, 2] += detected_center[0] - template_center[0]
    M_combined[1, 2] += detected_center[1] - template_center[1]
    
    return M_combined, detected_center, detected_radius

def overlay_dartboard_template(image: np.ndarray, 
                              reference_points: List[Tuple[int, int]],
                              template_color: Tuple[int, int, int] = (0, 255, 255),
                              template_thickness: int = 1,
                              show_numbers: bool = False,
                              show_analysis: bool = True) -> np.ndarray:
    """
    Overlay dartboard template on the image, fitted to reference points.
    """
    if len(reference_points) < 3:
        return image
    
    result = image.copy()
    
    try:
        # Create template
        template_radius = 400
        template = create_dartboard_template(template_radius)
        
        # Calculate transformation
        transform_matrix, center, radius = calculate_dartboard_transform(reference_points, template_radius)
        
        # Transform template to match dartboard
        transformed_template = cv2.warpAffine(template, transform_matrix, 
                                            (image.shape[1], image.shape[0]))
        
        # Create mask for template lines
        mask = cv2.cvtColor(transformed_template, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        
        # Draw template lines
        template_lines = np.where(mask > 0)
        result[template_lines] = template_color
        
        # Draw reference points and fitted circle
        cv2.circle(result, center, int(radius), (0, 255, 0), 2)
        for i, point in enumerate(reference_points):
            cv2.circle(result, point, 8, (0, 0, 255), -1)
            cv2.putText(result, str(i+1), (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show angle analysis
        if show_analysis:
            for i, point in enumerate(reference_points):
                dx = point[0] - center[0]
                dy = point[1] - center[1]
                angle_deg = math.degrees(math.atan2(dy, dx))
                if angle_deg < 0:
                    angle_deg += 360
                
                cv2.line(result, center, point, (255, 0, 255), 1)
                
                mid_x = center[0] + int(0.7 * dx)
                mid_y = center[1] + int(0.7 * dy)
                cv2.putText(result, f'{angle_deg:.0f}°', (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Add dartboard numbers
        if show_numbers:
            dartboard_numbers = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
            number_radius = int(radius * 0.9)
            
            # Get rotation angle from transformation matrix
            rotation_applied = math.atan2(transform_matrix[1, 0], transform_matrix[0, 0])
            
            for i, number in enumerate(dartboard_numbers):
                # Standard dartboard: 20 at top (-90°), then clockwise
                base_angle = (i * 18 - 90) * math.pi / 180
                final_angle = base_angle + rotation_applied
                
                x = int(center[0] + number_radius * math.cos(final_angle))
                y = int(center[1] + number_radius * math.sin(final_angle))
                
                cv2.putText(result, str(number), (x-8, y+4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(result, str(number), (x-8, y+4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Add info text
        info_text = f"Template fitted - Center: {center}, Radius: {radius:.1f}"
        cv2.putText(result, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    except Exception as e:
        print(f"Error overlaying template: {e}")
    
    return result

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
    
    # If we have enough points, fit template
    if len(corner_points) >= 3:
        # Create dartboard template overlay with analysis
        result = overlay_dartboard_template(image, corner_points, show_numbers=True, show_analysis=True)
        
        # Validate the fit
        is_valid = validate_dartboard_circle(corner_points)
        status_text = "Valid dartboard template" if is_valid else "Invalid template fit"
        cv2.putText(result, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_valid else (0, 0, 255), 2)
        
        # Add image filename to display
        filename_text = f"File: {os.path.basename(image_path)}"
        cv2.putText(result, filename_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display result
        cv2.imshow('Dartboard with Template Overlay', result)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return result
    else:
        print("Not enough corner points to fit template")
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

def get_all_sample_images(data_folder: str = "training/data/empty_board") -> List[Tuple[str, Optional[str]]]:
    """
    Get all sample images and their corresponding label files.
    
    Args:
        data_folder: Base path to the training data folder
    
    Returns:
        List of (image_path, label_path) tuples
    """
    good_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")
    
    # Find all image files in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(good_folder, ext)))
    
    if not image_files:
        print(f"No image files found in {good_folder}")
        return []
    
    # Sort images for consistent navigation
    image_files.sort()
    
    # Find corresponding label files for each image
    samples = []
    for image_path in image_files:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_file = os.path.join(labels_folder, f"{image_name}.txt")
        
        if os.path.exists(label_file):
            samples.append((image_path, label_file))
        else:
            samples.append((image_path, None))
            print(f"No label file found for {image_path}")
    
    return samples

def calculate_dart_score(point: Tuple[int, int], 
                        center: Tuple[int, int], 
                        radius: float,
                        rotation_angle: float = 0.0) -> Tuple[int, str]:
    """
    Calculate the dart score based on where it landed on the dartboard.
    
    Args:
        point: (x, y) coordinates of the dart hit
        center: (x, y) coordinates of the dartboard center
        radius: Radius of the dartboard
        rotation_angle: Rotation angle of the dartboard in radians
    
    Returns:
        (score, description) tuple
    """
    # Calculate distance from center (as a percentage of radius)
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    distance = np.sqrt(dx**2 + dy**2) / radius
    
    # Calculate angle relative to center, accounting for template rotation
    angle_rad = math.atan2(dy, dx) - rotation_angle
    angle_deg = math.degrees(angle_rad)
    
    # Normalize angle to 0-360 range
    if angle_deg < 0:
        angle_deg += 360
    
    # In our template, segment boundaries are at (i * 18 - 99)° and segments are 18° wide
    # Segment 20 is centered at -90° (270°), with boundaries at -99° (261°) and -81° (279°)
    # Adjust angle to account for this, shifting by 9° to align with segment boundaries
    adjusted_angle_deg = (angle_deg + 99) % 360
    
    # Determine segment number (0 to 19)
    segment_idx = int(adjusted_angle_deg / 18)
    dartboard_segments = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    segment_value = dartboard_segments[segment_idx % 20]
    
    # Determine multiplier based on distance from center
    if distance > 1.0:
        # Outside the dartboard
        return 0, "Miss"
    elif distance > 0.95:
        # Double ring
        return segment_value * 2, f"Double {segment_value}"
    elif distance > 0.625:
        # Outer single area
        return segment_value, f"Single {segment_value}"
    elif distance > 0.56:
        # Triple ring
        return segment_value * 3, f"Triple {segment_value}"
    elif distance > 0.10:
        # Inner single area
        return segment_value, f"Single {segment_value}"
    elif distance > 0.04:
        # Outer bull (25)
        return 25, "Outer Bull"
    else:
        # Inner bull (50)
        return 50, "Bull's Eye"

def predict_dart_score(event, x, y, flags, param):
    """
    Mouse callback function to predict dart score on click.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get parameters from param dictionary
        center = param['center']
        radius = param['radius']
        rotation = param.get('rotation', 0.0)
        image = param['image'].copy()
        
        # Draw dart position
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.line(image, center, (x, y), (0, 0, 255), 1, cv2.LINE_AA)
        
        # Calculate score
        score, description = calculate_dart_score((x, y), center, radius, rotation)
        
        # Display score on image
        score_text = f"Score: {score} - {description}"
        cv2.putText(image, score_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Update display
        cv2.imshow(param['window_name'], image)
        param['last_image'] = image
        param['last_score'] = (score, description)
        print(f"Dart at ({x}, {y}): {score} points - {description}")

def process_dartboard_images():
    """
    Process multiple dartboard images with keyboard navigation.
    """
    # Get all sample images and labels
    samples = get_all_sample_images()
    
    if not samples:
        print("No sample images found")
        return
    
    print(f"Found {len(samples)} sample images")
    
    # Start with the first image
    current_idx = 0
    window_name = 'Dartboard Template Testing'
    
    while True:
        # Get current image and label
        image_path, label_path = samples[current_idx]
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            current_idx = (current_idx + 1) % len(samples)
            continue
        
        print(f"\nProcessing image {current_idx+1}/{len(samples)}: {os.path.basename(image_path)}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Load labels if available
        corner_points = []
        if label_path:
            corner_points = load_yolo_labels(label_path, image.shape[1], image.shape[0])
            print(f"Loaded {len(corner_points)} corner points")
        
        # Process the image
        center = None
        radius = 0
        rotation = 0
        
        if len(corner_points) >= 3:
            # Create dartboard template overlay
            result = overlay_dartboard_template(image, corner_points, show_numbers=True, show_analysis=True)
            
            # Get dartboard parameters for scoring
            transform_matrix, center, radius = calculate_dartboard_transform(corner_points, 400)
            rotation = math.atan2(transform_matrix[1, 0], transform_matrix[0, 0])
            
            # Validate the fit
            is_valid = validate_dartboard_circle(corner_points)
            status_text = "Valid dartboard template" if is_valid else "Invalid template fit"
            cv2.putText(result, status_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_valid else (0, 0, 255), 2)
        else:
            print("Not enough corner points to fit template")
            # Just display the original image with any available points
            result = image.copy()
            for i, point in enumerate(corner_points):
                cv2.circle(result, point, 8, (0, 0, 255), -1)
                cv2.putText(result, str(i+1), (point[0]+10, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add image navigation info
        filename_text = f"File: {os.path.basename(image_path)} [{current_idx+1}/{len(samples)}]"
        cv2.putText(result, filename_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add score prediction instructions if dartboard is detected
        if center is not None:
            cv2.putText(result, "Click anywhere to predict dart score", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        nav_text = "Navigation: Right arrow = next, Left arrow = previous, ESC = exit"
        cv2.putText(result, nav_text, (10, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display result
        cv2.imshow(window_name, result)
        
        # Set up mouse callback for score prediction
        param = {
            'center': center, 
            'radius': radius, 
            'rotation': rotation,
            'image': result,
            'window_name': window_name,
            'last_image': result,
            'last_score': None
        }
        cv2.setMouseCallback(window_name, predict_dart_score, param)
        
        # Wait for key press
        while True:
            key = cv2.waitKey(100) & 0xFF
            
            # Navigation controls
            if key == 27:  # ESC key - exit
                cv2.destroyAllWindows()
                return
            elif key == 83 or key == ord('n') or key == ord('d'):  # Right arrow or 'n' or 'd' - next image
                current_idx = (current_idx + 1) % len(samples)
                break
            elif key == 81 or key == ord('p') or key == ord('a'):  # Left arrow or 'p' or 'a' - previous image
                current_idx = (current_idx - 1) % len(samples)
                break
            elif key == ord('r'):  # 'r' key - reset image (remove dart marks)
                cv2.imshow(window_name, result)
                param['last_image'] = result
                param['last_score'] = None
    
    cv2.destroyAllWindows()
    print("Image navigation ended")

# Update the main entry point to use the multi-image version
if __name__ == "__main__":
    # Process multiple dartboard images with navigation
    process_dartboard_images()
