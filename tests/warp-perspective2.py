import cv2
import numpy as np

i = 0

# 1) Bilder ladens
src  = cv2.imread("training/data/transferlearning/stg1/raw/00261.jpg")
ref  = cv2.imread("resources/dartboard-gerade.jpg")

#src = cv2.resize(src, (640, 480))
#ref = cv2.resize(ref, (640, 480))

# 2) Feature-Punkte finden & matchen (SIFT = hochwertige Features)
sift = cv2.SIFT_create(nfeatures=300)
kps1, des1 = sift.detectAndCompute(src, None)
kps2, des2 = sift.detectAndCompute(ref, None)

bf  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)[:]  # beste 80

# 3) Homographie schätzen
pts_src = np.float32([kps1[m.queryIdx].pt for m in matches])
pts_ref = np.float32([kps2[m.trainIdx].pt for m in matches])
H, _ = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 5.0)

# 4) Bild entzerren
h, w = ref.shape[:2]
warped = cv2.warpPerspective(src, H, (w, h))

test = cv2.drawMatches(src, kps1, ref, kps2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("src", src)
cv2.imshow("ref", ref)
cv2.imshow("warped", warped)


test = cv2.resize(test, (1280, 720))
cv2.imshow("matches", test)
cv2.waitKey(0)
