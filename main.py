import cv2
from pathlib import Path

from utils import camera, calibration, predict

DEBUG = True

if __name__ == "__main__":
    cam = camera.VideoStreamViewer(source=Path("training/data/transferlearning/stg1/raw"))
    cam.open_connection()
    if not cam.isOpened():
        print("Camera not opened")
        exit(1)

    calib = calibration.CameraCalibration(ref_img="resources/dartboard-gerade.jpg", debug=DEBUG)
    predictor = predict.Predictor(model_path="models/stg4.pt")

    frame = cv2.imread("training/data/transferlearning/stg1/raw/00001.jpg")
    if frame is None:
        print("Failed to grab frame")
        exit(1)

    # Perform initial calibration
    success, result = calib.initial_calibration(frame)
    if not success:
        print(f"Calibration failed: {result}")
        exit(1)

    cv2.imshow("Initial Frame", frame)
    cv2.imshow("Warped Image", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    i = 0
    while True:
        frame = cam.get_frame_raw()
        if frame is None:
            print("Failed to grab frame")
            break
        
        frame = calib.warp_frame(frame)
        if frame is None:
            print("Failed to warp frame")
            continue

        if DEBUG:
            cv2.imshow("Warped Frame", frame)

        results = predictor.predict(frame)
        if DEBUG:
            for result in results:
                annotated = result.plot()
                cv2.imshow("Annotated Frame", annotated)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(f"output/{i}.jpg", frame)
            print("Frame saved as output/frame.jpg")

        i += 1

    cam.release()
    cv2.destroyAllWindows()
