import cv2, numpy as np, time

REFRESH_SEC   = 0.1333            # Wie oft neu nach Kreisen suchen?
CAM_INDEX     = 1            # Deine Kamera
WIN_ORIG      = "Original + Box + Kreise"
WIN_CROP      = "Dart-Crop"

def detect_board(frame):
    """Sucht Kreise, gibt Bounding-Box + alle gefundenen Kreise zurück."""
    gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
    cv2.imshow("Gray", gray)                 # Debug-Anzeige
    cv2.waitKey(10)
    
    canny = cv2.Canny(gray, 128, 255)
    cv2.imshow("Canny", canny)               # Debug-Anzeige

    h, w = frame.shape[:2]
    circles = cv2.HoughCircles(
        canny, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=1000,
        param1=100, param2=30,              # <-- param2 runter!
        minRadius=int(min(h, w) * 0.25),
        maxRadius=int(min(h, w) * 0.6)
    )
    if circles is None:
        return None, None

    # circles = np.uint16(np.around(circles[0]))
    circles = np.round(circles[0]).astype(int)    # oder np.int32
    # größten Kreis herauspicken
    x, y, r = max(circles, key=lambda c: c[2])
    bbox = (max(0, x - r), max(0, y - r), min(w, x + r), min(h, y + r))
    return bbox, circles


cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Webcam nicht gefunden!")

bbox, circles, t_last = None, None, 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # ---------------- Bounding-Box + Kreise regelmäßig aktualisieren
    if bbox is None or time.time() - t_last > REFRESH_SEC:
        bbox, circles = detect_board(frame)
        t_last = time.time()

    # ---------------- Debug-Bild mit Kreisen zeichnen
    frame_draw = frame.copy()
    if circles is not None:
        # alle Kreise in Gelb
        for x, y, r in circles:
            cv2.circle(frame_draw, (x, y), r, (0, 255, 255), 2)  # gelbe Linie
            cv2.circle(frame_draw, (x, y), 3, (0, 255, 255), -1) # gelber Punkt

        # größten Kreis in Rot hervorheben
        x, y, r = max(circles, key=lambda c: c[2])
        cv2.circle(frame_draw, (x, y), r, (0, 0, 255), 3)
        cv2.circle(frame_draw, (x, y), 5, (0, 0, 255), -1)

    # ---------------- Bounding-Box + Crop
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
        crop = frame[y1:y2, x1:x2]
        cv2.imshow(WIN_CROP, crop)
    else:
        cv2.imshow(WIN_CROP, frame)

    cv2.imshow(WIN_ORIG, frame_draw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
