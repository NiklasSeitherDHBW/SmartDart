import numpy as np
import cv2
import math
from typing import List, Tuple, Optional


class DartboardScorePredictor:
    """
    A utility class for dartboard score prediction and template overlay.
    Handles dartboard detection, template fitting, and dart score calculation.
    """
    
    def __init__(self, template_radius: int = 400):
        """
        Initialize the dartboard score predictor.
        
        Args:
            template_radius: Radius of the dartboard template
        """
        self.template_radius = template_radius
        self.dartboard_numbers = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
        
        # Dartboard region ratios (as fraction of radius)
        self.outer_double = 1.0
        self.inner_double = 0.95
        self.outer_treble = 0.625
        self.inner_treble = 0.56
        self.outer_bull = 0.10
        self.inner_bull = 0.04
        
        # Current dartboard parameters
        self.center = None
        self.radius = 0
        self.rotation_angle = 0
        self.transform_matrix = None
        
    def fit_circle_through_points(self, points: List[Tuple[int, int]], 
                                 center_estimate: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[int, int], float]:
        """
        Fits a circle through dartboard corner points.
        
        Args:
            points: List of (x, y) coordinates of the corner points
            center_estimate: Optional estimate of dartboard center
        
        Returns:
            ((center_x, center_y), radius) of the fitted circle
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        points_array = np.array(points, dtype=np.float64)
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
    
    def validate_dartboard_circle(self, points: List[Tuple[int, int]], 
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
            center, radius = self.fit_circle_through_points(points)
            
            # Check if all points are within tolerance of the fitted circle
            for point in points:
                distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
                deviation = abs(distance - radius) / radius
                
                if deviation > tolerance_ratio:
                    return False
            
            return True
        
        except:
            return False
    
    def create_dartboard_template(self) -> np.ndarray:
        """
        Create a dartboard template with all field boundaries.
        
        Returns:
            Template image with dartboard fields drawn
        """
        # Create blank image (square with some padding)
        size = self.template_radius * 2 + 100
        template = np.zeros((size, size, 3), dtype=np.uint8)
        center = (size // 2, size // 2)
        
        # Calculate dartboard region radii
        outer_double = int(self.template_radius * self.outer_double)
        inner_double = int(self.template_radius * self.inner_double)
        outer_treble = int(self.template_radius * self.outer_treble)
        inner_treble = int(self.template_radius * self.inner_treble)
        outer_bull = int(self.template_radius * self.outer_bull)
        inner_bull = int(self.template_radius * self.inner_bull)
        
        # Draw concentric circles
        cv2.circle(template, center, outer_double, (255, 255, 255), 2)
        cv2.circle(template, center, inner_double, (255, 255, 255), 2)
        cv2.circle(template, center, outer_treble, (255, 255, 255), 2)
        cv2.circle(template, center, inner_treble, (255, 255, 255), 2)
        cv2.circle(template, center, outer_bull, (255, 255, 255), 2)
        cv2.circle(template, center, inner_bull, (255, 255, 255), 2)
        
        # Draw radial lines at segment boundaries
        for i in range(20):
            angle = (i * 18 - 99) * math.pi / 180  # Start at boundary, go clockwise
            x1 = int(center[0] + inner_bull * math.cos(angle))
            y1 = int(center[1] + inner_bull * math.sin(angle))
            x2 = int(center[0] + outer_double * math.cos(angle))
            y2 = int(center[1] + outer_double * math.sin(angle))
            cv2.line(template, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        return template
    
    def calculate_dartboard_transform(self, reference_points: List[Tuple[int, int]]) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """
        Calculate transformation matrix to fit template to detected dartboard.
        
        Args:
            reference_points: List of reference points on the dartboard
            
        Returns:
            (transform_matrix, center, radius) tuple
        """
        # Fit circle through reference points
        detected_center, detected_radius = self.fit_circle_through_points(reference_points)
        
        # Calculate scale factor
        scale = detected_radius / self.template_radius
        
        # Calculate rotation angle
        rotation_angle = 0
        
        if len(reference_points) >= 1:
            # Use the first reference point to estimate rotation
            first_point = reference_points[0]
            dx = first_point[0] - detected_center[0]
            dy = first_point[1] - detected_center[1]
            
            # Calculate angle of first reference point
            ref_angle = math.atan2(dy, dx)
            ref_angle_deg = math.degrees(ref_angle)
            if ref_angle_deg < 0:
                ref_angle_deg += 360
                
            # Find nearest segment boundary (every 18 degrees starting from -9)
            segment_angles = [(i * 18 - 9) % 360 for i in range(20)]
            nearest_segment = min(segment_angles, key=lambda x: min(abs(ref_angle_deg - x), 
                                                                  abs(ref_angle_deg - x + 360), 
                                                                  abs(ref_angle_deg - x - 360)))
            
            # Calculate rotation needed
            rotation_needed = nearest_segment - ref_angle_deg
            
            # Normalize to -180 to 180 range
            if rotation_needed > 180:
                rotation_needed -= 360
            elif rotation_needed < -180:
                rotation_needed += 360
                
            rotation_angle = math.radians(rotation_needed)
        
        # Create transformation matrix
        template_center = (self.template_radius + 50, self.template_radius + 50)
        
        # Combine scale, rotation, and translation
        M_combined = cv2.getRotationMatrix2D(template_center, math.degrees(rotation_angle), scale)
        M_combined[0, 2] += detected_center[0] - template_center[0]
        M_combined[1, 2] += detected_center[1] - template_center[1]
        
        return M_combined, detected_center, detected_radius
    
    def calibrate_dartboard(self, reference_points: List[Tuple[int, int]]) -> bool:
        """
        Calibrate the dartboard using reference points.
        
        Args:
            reference_points: List of reference points on the dartboard
            
        Returns:
            True if calibration was successful
        """
        if len(reference_points) < 3:
            return False
        
        try:
            # Calculate dartboard parameters
            self.transform_matrix, self.center, self.radius = self.calculate_dartboard_transform(reference_points)
            self.rotation_angle = math.atan2(self.transform_matrix[1, 0], self.transform_matrix[0, 0])
            return True
        except Exception as e:
            print(f"Dartboard calibration failed: {e}")
            return False
    
    def overlay_dartboard_template(self, image: np.ndarray, 
                                  reference_points: List[Tuple[int, int]] = None,
                                  template_color: Tuple[int, int, int] = (0, 255, 255),
                                  show_numbers: bool = False,
                                  show_analysis: bool = False) -> np.ndarray:
        """
        Overlay dartboard template on the image.
        
        Args:
            image: Input image
            reference_points: Optional reference points (if not already calibrated)
            template_color: Color for template lines (BGR)
            show_numbers: Whether to show dartboard numbers
            show_analysis: Whether to show angle analysis
            
        Returns:
            Image with dartboard template overlay
        """
        result = image.copy()
        
        # Use provided reference points or existing calibration
        if reference_points and len(reference_points) >= 3:
            if not self.calibrate_dartboard(reference_points):
                return result
        elif self.center is None:
            # No calibration available
            return result
        
        try:
            # Create template
            template = self.create_dartboard_template()
            
            # Transform template to match dartboard
            transformed_template = cv2.warpAffine(template, self.transform_matrix, 
                                                (image.shape[1], image.shape[0]))
            
            # Create mask for template lines
            mask = cv2.cvtColor(transformed_template, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            
            # Draw template lines
            template_lines = np.where(mask > 0)
            result[template_lines] = template_color
            
            # Draw center and radius
            cv2.circle(result, self.center, int(self.radius), (0, 255, 0), 2)
            cv2.circle(result, self.center, 5, (0, 255, 0), -1)
            
            # Draw reference points if provided
            if reference_points:
                for i, point in enumerate(reference_points):
                    cv2.circle(result, point, 8, (0, 0, 255), -1)
                    cv2.putText(result, str(i+1), (point[0]+10, point[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show angle analysis if requested
                if show_analysis:
                    for i, point in enumerate(reference_points):
                        dx = point[0] - self.center[0]
                        dy = point[1] - self.center[1]
                        angle_deg = math.degrees(math.atan2(dy, dx))
                        if angle_deg < 0:
                            angle_deg += 360
                        
                        cv2.line(result, self.center, point, (255, 0, 255), 1)
                        
                        mid_x = self.center[0] + int(0.7 * dx)
                        mid_y = self.center[1] + int(0.7 * dy)
                        cv2.putText(result, f'{angle_deg:.0f}°', (mid_x, mid_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # Add dartboard numbers if requested
            if show_numbers:
                number_radius = int(self.radius * 0.9)
                for i, number in enumerate(self.dartboard_numbers):
                    # Standard dartboard: 20 at top (-90°), then clockwise
                    base_angle = (i * 18 - 90) * math.pi / 180
                    final_angle = base_angle + self.rotation_angle
                    
                    x = int(self.center[0] + number_radius * math.cos(final_angle))
                    y = int(self.center[1] + number_radius * math.sin(final_angle))
                    
                    # Draw number with outline for better visibility
                    cv2.putText(result, str(number), (x-8, y+4), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(result, str(number), (x-8, y+4), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
        except Exception as e:
            print(f"Error overlaying template: {e}")
        
        return result
    
    def calculate_dart_score(self, dart_position: Tuple[int, int]) -> Tuple[int, str]:
        """
        Calculate the dart score based on where it landed on the dartboard.
        
        Args:
            dart_position: (x, y) coordinates of the dart hit
        
        Returns:
            (score, description) tuple
        """
        if self.center is None or self.radius == 0:
            return 0, "Dartboard not calibrated"
        
        # Calculate distance from center (as a percentage of radius)
        dx = dart_position[0] - self.center[0]
        dy = dart_position[1] - self.center[1]
        distance = np.sqrt(dx**2 + dy**2) / self.radius
        
        # Calculate angle relative to center, accounting for template rotation
        angle_rad = math.atan2(dy, dx) - self.rotation_angle
        angle_deg = math.degrees(angle_rad)
        
        # Normalize angle to 0-360 range
        if angle_deg < 0:
            angle_deg += 360
        
        # Adjust angle to account for dartboard orientation
        adjusted_angle_deg = (angle_deg + 99) % 360
        
        # Determine segment number (0 to 19)
        segment_idx = int(adjusted_angle_deg / 18)
        segment_value = self.dartboard_numbers[segment_idx % 20]
        
        # Determine multiplier based on distance from center
        if distance > 1.0:
            # Outside the dartboard
            return 0, "Miss"
        elif distance > self.inner_double:
            # Double ring
            return segment_value * 2, f"Double {segment_value}"
        elif distance > self.outer_treble:
            # Outer single area
            return segment_value, f"Single {segment_value}"
        elif distance > self.inner_treble:
            # Triple ring
            return segment_value * 3, f"Triple {segment_value}"
        elif distance > self.outer_bull:
            # Inner single area
            return segment_value, f"Single {segment_value}"
        elif distance > self.inner_bull:
            # Outer bull (25)
            return 25, "Outer Bull"
        else:
            # Inner bull (50)
            return 50, "Bull's Eye"
    
    def process_dart_detections(self, image: np.ndarray, 
                               dart_positions: List[Tuple[int, int]],
                               show_scores: bool = True) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
        """
        Process dart detections and calculate scores.
        
        Args:
            image: Input image
            dart_positions: List of (x, y) dart positions
            show_scores: Whether to draw scores on the image
            
        Returns:
            (processed_image, scores) tuple where scores is list of (score, description)
        """
        result = image.copy()
        dart_scores = []
        
        if self.center is None or self.radius == 0:
            return result, dart_scores
        
        # Process each dart
        for i, dart_pos in enumerate(dart_positions):
            # Calculate score for this dart
            score, description = self.calculate_dart_score(dart_pos)
            dart_scores.append((score, description))
            
            if show_scores:
                # Draw dart position
                dart_color = (0, 0, 255)  # Red for darts
                cv2.circle(result, dart_pos, 5, dart_color, -1)
                cv2.line(result, self.center, dart_pos, dart_color, 1, cv2.LINE_AA)
                
                # Show score near the dart
                score_text = f"{score}"
                text_pos = (dart_pos[0] + 10, dart_pos[1])
                cv2.putText(result, score_text, text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, dart_color, 2)
        
        # Display total score if multiple darts
        if dart_scores and show_scores:
            total_score = sum(score for score, _ in dart_scores)
            total_text = f"Total: {total_score}"
            cv2.putText(result, total_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result, dart_scores
    
    def is_calibrated(self) -> bool:
        """
        Check if the dartboard is calibrated and ready for score prediction.
        
        Returns:
            True if calibrated
        """
        return self.center is not None and self.radius > 0
    
    def get_dartboard_info(self) -> dict:
        """
        Get current dartboard calibration information.
        
        Returns:
            Dictionary with dartboard parameters
        """
        return {
            'center': self.center,
            'radius': self.radius,
            'rotation_angle': self.rotation_angle,
            'is_calibrated': self.is_calibrated()
        }
