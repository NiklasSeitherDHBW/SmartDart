#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dart-Tracker v1.3   (Bug-fix: Matrix-Multiplikation)
----------------------------------------------------
• automatische Kalibrierung (Ellipse-Fit, Auto-Rotation)
• perfekte Draufsicht, kompletter Zahlenring
• Live-Scoring jedes neu auftauchenden Pfeils
"""

import cv2
import numpy as np
import time
from math import sin, cos, radians, atan2, degrees, hypot

# ------------------------------ Board-Geometrie (mm)
R_BOARD_EDGE_MM  = 225.5
R_INNER_BULL_MM  =   6.35
R_OUTER_BULL_MM  =  15.9
R_TRIPLE_IN_MM   =  99.0
R_TRIPLE_OUT_MM  = 107.0
R_DOUBLE_IN_MM   = 162.0
R_DOUBLE_OUT_MM  = 170.0

SECTORS = [20,1,18,4,13,6,10,15,2,17,
           3,19,7,16,8,11,14,9,12,5]

# ------------------------------ Hyper-Parameter
CAM_INDEX        = 1
REFRESH_SEC      = 0.4
MARGIN           = 1.5          # 10 % Luft um das Board
THR_DIFF         = 40
MIN_LINE_LEN     = 60
LINE_CENTER_TOL  = 15

# ================================================== Hilfsfunktionen
def fit_outer_ellipse(frame):
    #gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(frame, (9, 9), 2)
    v = np.median(blur)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, lo, hi)

    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 50:
        return None
    return cv2.fitEllipse(cnt)      # ((cx,cy),(ma,mi),angle)

def ellipse_to_homography(ell, margin=MARGIN):
    (cx, cy), (ma, mi), theta = ell
    a, b   = (ma/2) * margin, (mi/2) * margin
    th     = radians(theta)
    r_px   = max(a, b)
    size   = int(2 * r_px * margin)

    phi = np.deg2rad(np.arange(0, 360, 72))
    src = np.float32([
        [cx + a*cos(p)*cos(th) - b*sin(p)*sin(th),
         cy + a*cos(p)*sin(th) + b*sin(p)*cos(th)]
        for p in phi
    ])
    dst = np.float32([
        [size/2 + r_px*margin*cos(p),
         size/2 + r_px*margin*sin(p)]
        for p in phi
    ])

    H, _ = cv2.findHomography(src, dst, 0)
    px2mm = R_BOARD_EDGE_MM / (r_px * margin)
    return H, size, px2mm

def detect_board_angle(warp):
    #gray  = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(warp, (5, 5), 2)
    cv2.imshow("Blur", blur)           # Debug-Anzeige
    v = np.median(blur)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, lo, hi) 
    
    cv2.imshow("Canny", edges)           # Debug-Anzeige
    
    h, w  = blur.shape[:2]
    cx, cy = w//2, h//2

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 120,
                            minLineLength=MIN_LINE_LEN,
                            maxLineGap=10)
    
    vis = warp.copy()
    best_len, best_ang = 0, 0

    if lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0]:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if (hypot(x1-cx, y1-cy) < LINE_CENTER_TOL or
                hypot(x2-cx, y2-cy) < LINE_CENTER_TOL):
                ang = degrees(atan2(y2 - y1, x2 - x1))
                length = hypot(x2 - x1, y2 - y1)
                if length > best_len:
                    best_len, best_ang = length, ang

    cv2.imshow("Detected Lines", vis)
    cv2.waitKey(1)

    return (best_ang)

def score_mm(x_mm, y_mm):
    r   = hypot(x_mm, y_mm)
    phi = (degrees(atan2(-y_mm, x_mm)) + 360) % 360
    sec = SECTORS[int(phi // 18)]

    if r < R_INNER_BULL_MM: return 50
    if r < R_OUTER_BULL_MM: return 25
    if R_TRIPLE_IN_MM < r < R_TRIPLE_OUT_MM: factor = 3
    elif R_DOUBLE_IN_MM < r < R_DOUBLE_OUT_MM: factor = 2
    else: factor = 1
    return sec * factor

# ================================================== Hauptschleife
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webcam nicht erreichbar – CAM_INDEX prüfen")

    H, ref, px2mm, size = None, None, 1.0, 0
    t_last = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # -------------------------- (Re-)Kalibrieren?
        if H is None and (time.time() - t_last > REFRESH_SEC):
            ellipse = fit_outer_ellipse(frame)
            t_last = time.time()

            if ellipse is not None:
                H_persp, size, px2mm = ellipse_to_homography(ellipse)

                # erste Warping-Stufe: perspektivisch entzerren
                warp = cv2.warpPerspective(frame, H_persp, (size, size))
                warp = frame
                # automatische Rotation
                rot = detect_board_angle(warp)
                M2  = cv2.getRotationMatrix2D((size/2, size/2), rot, 1.0)

                # ► Kombiniere Rotation & Perspektive korrekt
                R = np.eye(3, dtype=np.float32)   # 3 × 3-Identität
                R[:2, :] = M2                     # setze oberen Block = 2 × 3-Affine
                H = R @ H_persp                   # 3 × 3 • 3 × 3 = 3 × 3

                # Referenzbild mit beiden Transformationen erzeugen
                ref = cv2.warpPerspective(frame, H, (size, size))

        # -------------------------- Kein H → Live-Bild zeigen
        if H is None:
            cv2.imshow("Live", frame)
            if cv2.waitKey(1) == 27:          # ESC
                break
            continue

        # -------------------------- Entzerren + Rotieren in einem Schritt
        warp = cv2.warpPerspective(frame, H, (size, size))

        diff = cv2.absdiff(ref, warp)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, THR_DIFF, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 7)

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            (x_px, y_px), _ = cv2.minEnclosingCircle(c)

            x_mm = (x_px - size/2) * px2mm
            y_mm = (y_px - size/2) * px2mm
            pts  = score_mm(x_mm, y_mm)

            cv2.putText(warp, f"{pts}", (int(x_px), int(y_px)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            ref = warp.copy()       # Pfeil verschwindet für das nächste diff

        cv2.rectangle(warp, (0,0), (size-1, size-1), (0,255,0), 2)
        cv2.imshow("Live", frame)
        cv2.imshow("Warped", warp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:              # ESC
            break
        if key == ord('c'):        # „c“ = Re-Kalibrieren
            H = None

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------------------------- Start
if __name__ == "__main__":
    main()
