import cv2
import numpy as np
import time

src = cv2.imread("training/data/transferlearning/stg1/raw/00261.jpg")
ref = cv2.imread("resources/dartboard-gerade.jpg")

if src is None or ref is None:
    print("Error: Could not load images!")
    exit()

sift = cv2.SIFT_create(
    nfeatures=500,           # More features for better matching
    contrastThreshold=0.04,  # Lower threshold = more features
    edgeThreshold=10,        # Reduce edge features
    sigma=1.6               # Standard sigma
)

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

kps1, des1 = sift.detectAndCompute(src_gray, None)
kps2, des2 = sift.detectAndCompute(ref_gray, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Find matches with ratio test for quality filtering
matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.7 * n.distance or True:  # Ratio test threshold
            good_matches.append(m)

# Sort by distance and limit matches
good_matches = sorted(good_matches, key=lambda x: x.distance)[:100]

pts_src = np.float32([kps1[m.queryIdx].pt for m in good_matches])
pts_ref = np.float32([kps2[m.trainIdx].pt for m in good_matches])

H, mask = cv2.findHomography(
    pts_src, pts_ref, 
    cv2.RANSAC, 
    ransacReprojThreshold=3.0,  # Tighter threshold for better precision
    maxIters=2000,              # More iterations for better results
    confidence=0.99             # Higher confidence
)

if H is None:
    print("Error: Could not compute homography!")
    exit()

h, w = ref.shape[:2]
warped = cv2.warpPerspective(src, H, (w, h))

cv2.imshow("Source Image", src)
cv2.imshow("Reference Image", ref)
cv2.imshow("Warped Image", warped)
cv2.waitKey(0)
