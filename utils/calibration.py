import cv2
import numpy as np


class CameraCalibration:
    def __init__(self, debug=False):
        self.debug = debug

    def detect_board(self, frame):
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur = cv2.medianBlur(gray, 3)
        blur = cv2.GaussianBlur(frame, (9, 9), 2)
        v = np.median(blur)
        lo = int(max(0, 0.66 * v))
        hi = int(min(255, 1.33 * v))
        canny = cv2.Canny(blur, lo, hi)

        if self.debug:
            cv2.imshow("Gray", blur)
            cv2.imshow("Canny", canny)

        h, w = frame.shape[:2]
        circles = cv2.HoughCircles(
            canny, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=1000, param1=100, param2=30,
            minRadius=int(min(h, w) * 0.25),
            maxRadius=int(min(h, w) * 1)
        )

        if circles is None:
            return None, None

        circles = np.round(circles[0]).astype(int)
        x, y, r = max(circles, key=lambda c: c[2])
        bbox = (max(0, x - r), max(0, y - r), min(w, x + r), min(h, y + r))

        if self.debug:
            orig_frame = frame.copy()
            
            # Draw all circles in yellow
            for x, y, r in circles:
                cv2.circle(orig_frame, (x, y), r, (0, 255, 255), 2)

            # Highlight the largest circle in red
            cv2.circle(orig_frame, (x, y), r, (0, 0, 255), 3)

            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cropped_frame = frame[y1:y2, x1:x2]
                cv2.imshow("Cropped Frame", cropped_frame)
            else:
                cv2.imshow("Cropped Frame", frame)
                
            cv2.imshow("Detected Circles", orig_frame)

        return bbox, circles
    
