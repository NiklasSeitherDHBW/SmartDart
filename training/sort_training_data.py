import cv2
import os
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import camera, calibration, predict


if __name__ == "__main__":
    root_dir = Path("training/data/transferlearning/stg3/raw")
    print(root_dir.absolute())
    existing_files = [root_dir / f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    i = 0
    print(existing_files[i])
    
    cam = camera.VideoStreamViewer(source=existing_files[i])
    cam.open_connection()
    if not cam.isOpened():
        print("Camera not opened")
        exit(1)

    calibrator = calibration.CameraCalibration(ref_img="resources/dartboard-gerade.jpg", debug=True)
    predictor = predict.Predictor(model_path="training/runs/train/Yolo8n-finetune/weights/best.pt")

    window_title = "Darts Annotation"  # Consistent window name
    processed = False

    frame = cam.get_frame_raw()
    if frame is None:
        print("Failed to grab frame")
        exit(1)
    
    while True:
        # Get just the filename to display as text
        current_filename = existing_files[i].name
        display_frame = frame.copy()

        #display_frame = display_frame[32:591, 340:835]
        
        display_frame = calibrator.warp_frame(display_frame)
        display_frame = cv2.resize(display_frame, (800, 800))
        orig_frame_cropped = display_frame.copy()

        if not processed:
            # Predict and annotate the frame
            results = predictor.predict(display_frame)
            if results:
                display_frame = results[0].plot()

                # Add filename as text on the annotated frame
                cv2.putText(display_frame, current_filename, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(window_title, display_frame)
                processed = True 
            else:
                print("No results found, skipping annotation.")
                # Add filename as text on the original frame
                cv2.putText(display_frame, current_filename, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow(window_title, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord("g"):
            # Save the current frame with the filename
            filename = existing_files[i]
            saved_path = str(filename).replace("raw", "good")
            cv2.imwrite(saved_path, orig_frame_cropped)
            print(f"Frame saved as '{saved_path}'")

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
            processed = False

        if key == ord("o"):
            # Save the current frame with the filename
            filename = existing_files[i]
            saved_path = str(filename).replace("raw", "okay")
            cv2.imwrite(saved_path, orig_frame_cropped)
            print(f"Frame saved as '{saved_path}'")

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
            processed = False

        if key == ord('b'):
            filename = existing_files[i]
            saved_path = str(filename).replace("raw", "bad")
            cv2.imwrite(saved_path, orig_frame_cropped)
            print(f"Frame saved as '{saved_path}'")
            
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
            processed = False
        
        if key == ord('c'):
            processed = False

    cam.release()
    cv2.destroyAllWindows()