import cv2
import os
from utils import camera, calibration, predict
from pathlib import Path

if __name__ == "__main__":
    root_dir = Path("data/train/raw")
    print(root_dir.absolute())
    existing_files = [root_dir / f for f in os.listdir(root_dir) if f.endswith('.jpg')]
    
    print(existing_files[0])
    
    cam = camera.VideoStreamViewer(source=existing_files[0])
    cam.open_connection()
    if not cam.isOpened():
        print("Camera not opened")
        exit(1)
        
    predictor = predict.Predictor(model_path="models/yolo8n.pt")

    i = 0
    processed = False
    window_title = "Darts Annotation"  # Consistent window name
    
    # Create window once
    cv2.namedWindow(window_title)
    
    frame = cam.get_frame_raw()
    if frame is None:
        print("Failed to grab frame")
        exit(1)
    
    while True:
        # Get just the filename to display as text
        current_filename = existing_files[i].name
        display_frame = frame.copy()
        
        if not processed:
            # Predict and annotate the frame
            results = predictor.predict(frame)
            if results:
                for result in results:
                    display_frame = result.plot()
                    # Add filename as text on the annotated frame
                cv2.putText(display_frame, current_filename, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(window_title, display_frame)
                processed = True
            else:
                print("No results found, skipping annotation.")
                # Add filename as text on the original frame
                cv2.putText(display_frame, current_filename, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(window_title, display_frame)
        else:
            # Add filename as text
            cv2.putText(display_frame, current_filename, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window_title, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            processed = False
            i += 1
            if i >= len(existing_files):
                print("All files processed")
                break
            # Update camera source to next file
            cam.release()
            cam = camera.VideoStreamViewer(source=existing_files[i])
            cam.open_connection()
            if not cam.isOpened():
                print(f"Failed to open {existing_files[i]}")
                break
            frame = cam.get_frame_raw()
            if frame is None:
                print("Failed to grab frame")
                break

    cam.release()
    cv2.destroyAllWindows()