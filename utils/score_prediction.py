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
        self.dartboard_numbers = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                                  3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
        
        # Dartboard region ratios (as fraction of radius)
        self.outer_double = 1.0
        self.inner_double = 0.95
        self.outer_treble = 0.625
        self.inner_treble = 0.56
        self.outer_bull = 0.10
        self.inner_bull = 0.04
        
        # Calibration parameters
        self.center: Optional[Tuple[float, float]] = None
        self.radius: float = 0.0
        self.rotation_angle: float = 0.0
        self.transform_matrix: Optional[np.ndarray] = None
        
    def fit_circle_through_points(self, points: List[Tuple[int, int]], 
                                 center_estimate: Optional[Tuple[float, float]] = None
                                 ) -> Tuple[Tuple[float, float], float]:
        """
        Fits a circle through dartboard corner points.
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        pts = np.array(points, dtype=np.float64)
        x, y = pts[:,0], pts[:,1]
        A = np.column_stack((x, y, np.ones_like(x)))
        b = -(x**2 + y**2)
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        D, E, F = coeffs
        cx = -D/2
        cy = -E/2
        r = math.sqrt((D*D + E*E)/4 - F)
        return (cx, cy), r

    def create_dartboard_template(self) -> np.ndarray:
        """
        Create a dartboard template with all field boundaries.
        """
        size = 2 * self.template_radius + 100
        template = np.zeros((size, size, 3), dtype=np.uint8)
        center = (size//2, size//2)
        # Radii
        R_outD = int(self.template_radius * self.outer_double)
        R_inD  = int(self.template_radius * self.inner_double)
        R_outT = int(self.template_radius * self.outer_treble)
        R_inT  = int(self.template_radius * self.inner_treble)
        R_outB = int(self.template_radius * self.outer_bull)
        R_inB  = int(self.template_radius * self.inner_bull)
        # Circles
        cv2.circle(template, center, R_outD, (255,255,255), 2)
        cv2.circle(template, center, R_inD,  (255,255,255), 2)
        cv2.circle(template, center, R_outT, (255,255,255), 2)
        cv2.circle(template, center, R_inT,  (255,255,255), 2)
        cv2.circle(template, center, R_outB, (255,255,255), 2)
        cv2.circle(template, center, R_inB,  (255,255,255), 2)
        # Radial lines
        for i in range(20):
            angle = math.radians(i*18 - 9)
            x1 = int(center[0] + R_inB * math.cos(angle))
            y1 = int(center[1] + R_inB * math.sin(angle))
            x2 = int(center[0] + R_outD * math.cos(angle))
            y2 = int(center[1] + R_outD * math.sin(angle))
            cv2.line(template, (x1,y1), (x2,y2), (255,255,255), 1)
        return template

    def calculate_dartboard_transform(self, reference_points: List[Tuple[int,int]]):
        """
        Calculate transformation matrix to fit template to detected board.
        """
        center, radius = self.fit_circle_through_points(reference_points)
        scale = radius / self.template_radius
        # Rotation: align first point
        rot = 0.0
        if reference_points:
            dx = reference_points[0][0] - center[0]
            dy = reference_points[0][1] - center[1]
            ang = math.degrees(math.atan2(dy, dx)) % 360
            seg_angles = [(i*18 - 9) % 360 for i in range(20)]
            nearest = min(seg_angles, key=lambda x: min(abs(x-ang),360-abs(x-ang)))
            delta = nearest - ang
            if delta > 180: delta -= 360
            if delta < -180: delta += 360
            rot = math.radians(delta)
        size = 2*self.template_radius + 100
        tpl_ctr = (size/2, size/2)
        M = cv2.getRotationMatrix2D(tuple(tpl_ctr), math.degrees(rot), scale)
        M[0,2] += center[0] - tpl_ctr[0]
        M[1,2] += center[1] - tpl_ctr[1]
        return M, center, radius

    def calibrate_dartboard(self, reference_points: List[Tuple[int,int]]) -> bool:
        if len(reference_points) < 3:
            return False
        M, ctr, rad = self.calculate_dartboard_transform(reference_points)
        self.transform_matrix = M
        self.center = ctr
        self.radius = rad
        # Derive rotation_angle from M
        self.rotation_angle = math.atan2(M[1,0], M[0,0])
        return True

    def overlay_dartboard_template(self, image: np.ndarray,
                                  reference_points: Optional[List[Tuple[int,int]]] = None,
                                  template_color: Tuple[int,int,int] = (0,255,255),
                                  show_numbers: bool = False,
                                  show_analysis: bool = False) -> np.ndarray:
        out = image.copy()
        if reference_points and not self.is_calibrated():
            self.calibrate_dartboard(reference_points)
        if self.center is None:
            return out
        tpl = self.create_dartboard_template()
        warp = cv2.warpAffine(tpl, self.transform_matrix, (image.shape[1], image.shape[0]))
        mask = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Dartboard Template", warp)
        _,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        ys,xs = np.where(mask>0)
        out[ys,xs] = template_color
        # Draw board circle
        cx,cy = int(self.center[0]), int(self.center[1])
        cv2.circle(out, (cx,cy), int(self.radius), (0,255,0),2)
        cv2.circle(out, (cx,cy), 5, (0,255,0),-1)
        # Analysis
        if reference_points and show_analysis:
            for i,pt in enumerate(reference_points):
                dx,dy = pt[0]-self.center[0], pt[1]-self.center[1]
                ang = math.degrees(math.atan2(dy,dx))%360
                cv2.line(out, (cx,cy), pt, (255,0,255),1)
                mx, my = int(cx+0.7*dx), int(cy+0.7*dy)
                cv2.putText(out, f"{ang:.0f}°", (mx,my), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255),1)
        # Numbers
        if show_numbers:
            num_r = int(self.radius*0.9)
            for i,val in enumerate(self.dartboard_numbers):
                ang = math.radians(i*18 - 90) + self.rotation_angle
                x = int(self.center[0] + num_r*math.cos(ang))
                y = int(self.center[1] + num_r*math.sin(ang))
                cv2.putText(out, str(val), (x-8,y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),3)
                cv2.putText(out, str(val), (x-8,y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),2)
        return out

    def calculate_dart_score(self, dart_position: Tuple[int,int]) -> Tuple[int,str]:
        if self.center is None or self.radius==0:
            return 0, "Not calibrated"
        dx = dart_position[0] - self.center[0]
        dy = dart_position[1] - self.center[1]
        dist = math.hypot(dx,dy) / self.radius
        # Calculate angle from dart position relative to dartboard center
        # Add 90 degrees to align with the coordinate system where segment 20 is at top
        ang = math.degrees(math.atan2(dy,dx)) + 90 - math.degrees(self.rotation_angle)
        ang %= 360
        # Align so segment 20 is at top (adjust for dartboard segment layout)
        adj = (ang + 9) % 360
        idx = int(adj // 18)
        val = self.dartboard_numbers[idx]
        if dist>1.0:
            return 0, "Miss"
        if dist>self.inner_double:
            return val*2, f"Double {val}"
        if dist>self.outer_treble:
            return val, f"Single {val}"
        if dist>self.inner_treble:
            return val*3, f"Triple {val}"
        if dist>self.outer_bull:
            return val, f"Single {val}"
        if dist>self.inner_bull:
            return 25, "Outer Bull"
        return 50, "Bull's Eye"

    def is_calibrated(self) -> bool:
        """
        Check if the dartboard is calibrated and ready for score prediction.
        """
        return self.center is not None and self.radius > 0

    def process_dart_detections(self, image: np.ndarray,
                               dart_positions: List[Tuple[int,int]],
                               show_scores: bool = True):
        out = image.copy()
        scores = []
        if not self.is_calibrated():
            return out, scores
        for pt in dart_positions:
            sc,desc = self.calculate_dart_score(pt)
            scores.append((sc,desc))
            if show_scores:
                c = (0,0,255)
                cv2.circle(out, pt,5,c,-1)
                cv2.line(out, (int(self.center[0]),int(self.center[1])), pt, c,1)
                cv2.putText(out, str(sc), (pt[0]+10,pt[1]), cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
        if scores and show_scores:
            tot = sum(s for s,_ in scores)
            cv2.putText(out, f"Total: {tot}", (10,120), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        return out, scores

    def get_dartboard_info(self) -> dict:
        return {
            'center': self.center,
            'radius': self.radius,
            'rotation_angle': self.rotation_angle,
            'is_calibrated': self.is_calibrated()
        }
