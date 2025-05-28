import cv2
import numpy as np

# Load reference image once
ref = cv2.imread("data/dartboard-gerade.jpg")

# Initialize webcam
cap = cv2.VideoCapture(1)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Press 'q' to quit")
ret, frame = cap.read()
while True:
    # Capture frame from webcam
    
    if not ret:
        print("Error: Failed to capture image")
        break
    
    # Use current frame as source image
    src = frame
    
    # Feature detection and matching
    orb = cv2.ORB_create(1000)
    kps1, des1 = orb.detectAndCompute(src, None)
    kps2, des2 = orb.detectAndCompute(ref, None)
    
    # Skip processing if no features found
    if des1 is None or des2 is None:
        cv2.imshow("Original", src)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Need at least 4 matches for homography
    if len(matches) > 4:
        matches = sorted(matches, key=lambda x: x.distance)[:80]  # best 80
        
        # Estimate homography
        pts_src = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts_ref = np.float32([kps2[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 5.0)
        
        # Warp perspective
        h, w = ref.shape[:2]
        warped = cv2.warpPerspective(src, H, (w, h))
        
        # Display results
        cv2.imshow("Original", src)
        cv2.imshow("Warped", warped)
    else:
        # If not enough matches, just show original
        cv2.imshow("Original", src)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
