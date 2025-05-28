import cv2
import numpy as np

i = 4

# 1) Bilder ladens
src  = cv2.imread(r"data\train\raw\00261.jpg")
ref  = cv2.imread("data/dartboard-gerade.jpg")

# 2) Feature-Punkte finden & matchen (ORB = schnell & patentfrei)
orb = cv2.ORB_create(1000)
kps1, des1 = orb.detectAndCompute(src, None)
kps2, des2 = orb.detectAndCompute(ref, None)

bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)[:80]  # beste 80

# 3) Homographie schätzen
pts_src = np.float32([kps1[m.queryIdx].pt for m in matches])
pts_ref = np.float32([kps2[m.trainIdx].pt for m in matches])
H, _ = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 5.0)

# 4) Bild entzerren
h, w = ref.shape[:2]
warped = cv2.warpPerspective(src, H, (w, h))

cv2.imwrite(f"dart_entzerrt_{i}.jpg", warped)
