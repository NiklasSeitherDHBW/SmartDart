import cv2
import numpy as np
from pathlib import Path


class VideoStreamViewer:
    def __init__(self, source=0, target_height=None, target_width=None):
        self.source = source
        self.height = target_height
        self.width = target_width
        self.cap = None

        self.source_is_folder = isinstance(source, Path)
        self.images = None


    def open_connection(self):
        if self.source_is_folder:
            # If source is a folder, use the first image in the folder
            self.images = list(Path(self.source).glob("*.jpg")) + list(Path(self.source).glob("*.png"))
            if not self.images:
                raise ValueError(f"No images found in folder: {self.source}")
        else:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

            # Set manual exposure mode and reduce exposure value significantly
            # Auto exposure mode = 0.75 (auto), 0.25 (manual)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Set to manual mode
            # self.cap.set(cv2.CAP_PROP_EXPOSURE, -7.5)  # Negative values mean shorter exposure time
            
            # Reduce gain if available
            self.cap.set(cv2.CAP_PROP_GAIN, 0)
            
            self.cap.set(cv2.CAP_PROP_FPS, 1)
            self.cap.set(cv2.CAP_PROP_GAMMA, 0.7)  # Lower gamma helps with overexposure

    def isOpened(self):
        if self.source_is_folder:
            return len(self.images) > 0
        return self.cap.isOpened()

    def release(self):
        if self.source_is_folder:
            self.images = None
            return None
        return self.cap.release()

    def get_frame_raw(self):
        """Capture a frame from the video stream."""
        if self.source_is_folder:
            if not self.images:
                print("No images available in the folder.")
                return None
            
            # Read the next image from the list
            img_path = self.images.pop(0)
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Failed to read image: {img_path}")
                return None
            
            # Resize if target dimensions are specified
            if self.height and self.width:
                frame = cv2.resize(frame, (self.width, self.height))

        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Failed to grab frame")
                return None

        return frame
