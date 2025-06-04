import cv2
from pathlib import Path

from utils import camera, calibration, predict, score_prediction

DEBUG = True

if __name__ == "__main__":
    cam = camera.VideoStreamViewer(source=Path("training/data/transferlearning/stg1/raw"))
    cam.open_connection()
    if not cam.isOpened():
        print("Camera not opened")
        exit(1)

    calib = calibration.CameraCalibration(ref_img="resources/dartboard-gerade.jpg", debug=DEBUG)
    predictor = predict.Predictor(model_path="models/stg4.pt")
    
    # Initialize dartboard score predictor
    score_predictor = score_prediction.DartboardScorePredictor()

    frame = cv2.imread("training/data/transferlearning/stg1/raw/00251.jpg")
    if frame is None:
        print("Failed to grab frame")
        exit(1)

    # Perform initial calibration
    success, result = calib.initial_calibration(frame)
    if not success:
        print(f"Calibration failed: {result}")
        exit(1)

    cv2.imshow("Initial Frame", cv2.resize(frame, (640, 480)))
    cv2.imshow("Warped Image", cv2.resize(result, (640, 480)))

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

        # Run YOLO prediction
        results = predictor.predict(frame)
        
        # Extract dart positions from YOLO results
        dart_positions = []
        dartboard_points = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box center
                    x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                    
                    # Check class ID to differentiate between darts and dartboard features
                    class_id = int(box.cls[0])
                    
                    if class_id == 4:  # Assuming class 4 is darts
                        dart_positions.append((x_center, y_center))
                    else:
                        # Collect other detected features as potential dartboard reference points
                        dartboard_points.append((x_center, y_center))
        
        # Try to calibrate dartboard if we have enough reference points
        if len(dartboard_points) >= 3 and not score_predictor.is_calibrated():
            if score_predictor.calibrate_dartboard(dartboard_points):
                print("Dartboard calibrated successfully!")
                print(f"Dartboard info: {score_predictor.get_dartboard_info()}")
        
        # Create display frame with dartboard template overlay
        display_frame = frame.copy()
        
        # Overlay dartboard template if calibrated
        if score_predictor.is_calibrated():
            display_frame = score_predictor.overlay_dartboard_template(
                display_frame, 
                show_numbers=True,
                template_color=(0, 255, 255)
            )
            
            # Process dart detections and calculate scores
            if dart_positions:
                display_frame, dart_scores = score_predictor.process_dart_detections(
                    display_frame, 
                    dart_positions, 
                    show_scores=True
                )
                
                # Print scores to console
                if dart_scores:
                    print(f"Frame {i}: Detected {len(dart_scores)} darts")
                    for j, (score, description) in enumerate(dart_scores):
                        print(f"  Dart {j+1}: {score} points - {description}")
                    total_score = sum(score for score, _ in dart_scores)
                    print(f"  Total Score: {total_score}")
        
        if DEBUG:
            # Show original YOLO annotations
            for result in results:
                annotated = result.plot()
                cv2.imshow("YOLO Detections", annotated)
            
            # Show dartboard analysis
            cv2.imshow("Dartboard Analysis", display_frame)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(f"output/{i}.jpg", display_frame)
            print(f"Frame saved as output/{i}.jpg")
        if key == ord('c'):
            # Manual calibration - collect dartboard reference points from YOLO detections
            if len(dartboard_points) >= 3:
                score_predictor.calibrate_dartboard(dartboard_points)
                print("Manual calibration completed!")

        i += 1

    cam.release()
    cv2.destroyAllWindows()
