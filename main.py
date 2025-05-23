import cv2
import numpy as np

from utils import camera, calibration, predict

DEBUG = True

if __name__ == "__main__":
    template = cv2.imread('data/dartboard.jpg', 0)

    cam = camera.VideoStreamViewer(source=1)
    cam.open_connection()
    if not cam.isOpened():
        print("Camera not opened")
        exit(1)
        
    calib = calibration.CameraCalibration(debug=DEBUG)
    predictor = predict.Predictor(model_path="models/yolo8n.pt")
    
    img = cam.get_frame_raw()
    if img is None:
        print("Failed to grab frame")
        exit(1)
    
    bbox, circles = calib.detect_board(img)
    x1, y1, x2, y2 = bbox
    
    while True:
        frame = cam.get_frame_raw()
        if img is None:
            print("Failed to grab frame")
            break
        
        cropped_frame = frame[y1:y2, x1:x2]
        results = predictor.predict(cropped_frame)
        
        if DEBUG:
            for result in results:
                annotated = result.plot()
                cv2.imshow("Annotated Frame", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
