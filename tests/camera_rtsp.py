import cv2
import numpy as np


def resize_frame_fixed(frame, width, height):
    # Get original dimensions
    h, w = frame.shape[:2]
    # Compute scale factor and new size
    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h))

    # Create a black image and paste the resized frame centered
    new_frame = np.zeros((height, width, 3), dtype=np.uint8)
    y_offset = (height - new_h) // 2
    x_offset = (width - new_w) // 2
    new_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result


def resize_frame_accordingly(frame, width):
    # Get original dimensions
    h, w = frame.shape[:2]
    # Compute scale factor and new size
    scale = width / w
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h))

    return resized

target_width = 800
target_height = 600
vcap = cv2.VideoCapture("rtsp://172.16.1.1:7447/diMrdwy9TotwtCxb")

# 3480x2160: dsxtKm0RPxuj6gUJ
# 1280x720: diMrdwy9TotwtCxb
# 640x360: jESFoDodeO8wqfQ0

if not vcap.isOpened():
    print("Error: Cannot open RTSP stream")
    exit()

while True:
    ret, frame = vcap.read()
    if not ret or frame is None:
        print("Failed to grab frame")
        break
    
    # result = resize_frame_fixed(frame, target_width, target_height)
    result = resize_frame_accordingly(frame, target_width)

    cv2.imshow('VIDEO', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vcap.release()
cv2.destroyAllWindows()
