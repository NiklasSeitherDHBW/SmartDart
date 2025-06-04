import cv2
import os
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import camera, calibration, predict
import numpy as np


def load_yolo_points(label_path, img_width, img_height, ref_classes=[0,1,2,3,5,6]):
    """Lädt die Referenzpunkte aus dem Label (YOLO-Format) und gibt sie als Pixelkoordinaten zurück."""
    points = []
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return points
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                class_id = int(parts[0])
                if class_id in ref_classes:
                    x = float(parts[1]) * img_width
                    y = float(parts[2]) * img_height
                    points.append([x, y])
    return np.array(points, dtype=np.float32)


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
    predictor = predict.Predictor(model_path="models/stg4.pt")

    window_title = "Darts Annotation"  # Consistent window name
    processed = False

    frame = cam.get_frame_raw()
    if frame is None:
        print("Failed to grab frame")
        exit(1)
    
    # --- Schablonen-Overlay vorbereiten ---
    # Schablone laden (als PNG mit Transparenz, z.B. resources/dartboard_template.png)
    template_path = Path("resources/dartboard_template.png")
    if not template_path.exists():
        print(f"Schablonenbild nicht gefunden: {template_path}")
        template_overlay = None
    else:
        template_overlay = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)  # PNG mit Alpha

    # --- Label für das aktuelle Bild laden ---
    label_path = Path(str(existing_files[i]).replace("raw", "labels").replace(".jpg", ".txt"))
    img_height, img_width = frame.shape[:2]
    ref_points = load_yolo_points(label_path, img_width, img_height)

    # Zielpunkte auf der Schablone (müssen zu den Referenzpunkten passen, z.B. Kreisrandpunkte)
    # Beispiel: Wenn die Schablone 800x800 ist und die Punkte im Uhrzeigersinn liegen:
    # (Diese Werte musst du ggf. anpassen, je nachdem wie die Schablone aufgebaut ist)
    template_h, template_w = 800, 800
    template_points = np.array([
        [template_w*0.43, template_h*0.12],  # Punkt 0
        [template_w*0.57, template_h*0.87],  # Punkt 1
        [template_w*0.12, template_h*0.57],  # Punkt 2
        [template_w*0.88, template_h*0.43],  # Punkt 3
        [template_w*0.15, template_h*0.34],  # Punkt 5
        [template_w*0.85, template_h*0.66],  # Punkt 6
    ], dtype=np.float32)

    # Die Reihenfolge und Anzahl der Punkte muss mit den Referenzpunkten übereinstimmen!

    while True:
        # Get just the filename to display as text
        current_filename = existing_files[i].name
        display_frame = frame.copy()

        #display_frame = display_frame[32:591, 340:835]
        
        display_frame = calibrator.warp_frame(display_frame)
        display_frame = cv2.resize(display_frame, (800, 800))
        orig_frame_cropped = display_frame.copy()

        # --- Schablonen-Overlay anwenden ---
        if template_overlay is not None and len(ref_points) == len(template_points):
            # Homographie berechnen
            H, _ = cv2.findHomography(template_points, ref_points)
            # Schablone auf das Bild mappen
            overlay_warped = cv2.warpPerspective(template_overlay, H, (display_frame.shape[1], display_frame.shape[0]))
            # Overlay mit Alpha-Kanal einblenden
            if overlay_warped.shape[2] == 4:
                alpha = overlay_warped[:,:,3] / 255.0
                for c in range(3):
                    display_frame[:,:,c] = (1-alpha) * display_frame[:,:,c] + alpha * overlay_warped[:,:,c]
                display_frame = display_frame.astype(np.uint8)

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