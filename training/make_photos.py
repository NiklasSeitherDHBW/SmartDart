import cv2
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import camera, calibration, predict

DEBUG = True

if __name__ == "__main__":
    cam = camera.VideoStreamViewer(source=1)
    cam.open_connection()
    if not cam.isOpened():
        print("Camera not opened")
        exit(1)

    img = cam.get_frame_raw()
    if img is None:
        print("Failed to grab frame")
        exit(1)

    while True:
        frame = cam.get_frame_raw()
        if frame is None:
            print("Failed to grab frame")
            break

        cropped_frame = frame#[y1:y2, x1:x2]
        cv2.imshow("Cropped Frame", cropped_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            existing_files = [f for f in os.listdir("data/train/raw") if f.endswith('.jpg')]
            indices = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
            next_index = max(indices) + 1 if indices else 1
            
            # pad the index with leading zeros to ensure consistent naming
            next_index = str(next_index).zfill(5)
            
            filename = f"data/train/raw/{next_index}.jpg"
            
            cv2.imwrite(filename, cropped_frame)
            print(f"Frame saved as '{filename}'")
            
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
