import cv2
import numpy as np


class VideoStreamViewer:
    def __init__(self, source=0, target_height=None, target_width=None):
        self.source = source
        self.height = target_height
        self.width = target_width
        self.cap = None

    def open_connection(self):
        self.cap = cv2.VideoCapture(self.source)
        
    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        return self.cap.release()

    def resize_frame_fixed(self, frame):
        """Resize and pad the frame to fixed width and height (letterbox)."""
        # Get original dimensions
        h, w = frame.shape[:2]
        # Compute scale factor and new size
        scale = min(self.width / w, self.height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))

        # Create a black image and paste the resized frame centered
        new_frame = np.zeros(
            (self.height, self.width, 3), dtype=np.uint8)
        y_offset = (self.height - new_h) // 2
        x_offset = (self.width - new_w) // 2
        new_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return new_frame

    def resize_frame_keep_ratio(self, frame):
        """Resize frame to width, keep aspect ratio."""
        h, w = frame.shape[:2]
        scale = self.width / w
        new_h = int(h * scale)
        resized = cv2.resize(frame, (self.width, new_h))
        return resized

    def get_frame_raw(self):
        """Capture a frame from the video stream."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            return None

        return frame

    def get_frame_resized(self):
        """Capture a frame and resize it."""
        frame = self.get_frame_raw()
        if frame is None:
            return None

        if self.height is not None and self.width is not None:
            return self.resize_frame_fixed(frame)
        elif self.width is not None:
            return self.resize_frame_keep_ratio(frame)
        else:
            return frame
