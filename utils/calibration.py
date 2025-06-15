import cv2
import numpy as np


class CameraCalibration:
    def __init__(self, ref_img, debug=False):
        self.debug = debug
        if ref_img is not None:
            self.ref_img = cv2.imread(ref_img)
            if self.ref_img is None:
                raise ValueError(f"Could not load reference image: {ref_img}")

        self.sift = cv2.SIFT_create(
            nfeatures=500,           # More features for better matching
            contrastThreshold=0.04,  # Lower threshold = more features
            edgeThreshold=10,        # Reduce edge features
            sigma=1.6               # Standard sigma
        )

        self.H = None


    def initial_calibration(self, frame):
        self.H = None

        src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)

        kps1, des1 = self.sift.detectAndCompute(src_gray, None)
        kps2, des2 = self.sift.detectAndCompute(ref_gray, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

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

        self.H, mask = cv2.findHomography(
            pts_src, pts_ref, 
            cv2.RANSAC, 
            ransacReprojThreshold=3.0,  # Tighter threshold for better precision
            maxIters=2000,              # More iterations for better results
            confidence=0.99             # Higher confidence
        )

        if self.H is None:
            return False, "Could not compute homography!"
        
        if self.debug:
            # Zeige SIFT Keypoints als Kreise
            img1_with_keypoints = cv2.drawKeypoints(frame, kps1, None, 
                                                   color=(0, 255, 0), 
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img2_with_keypoints = cv2.drawKeypoints(self.ref_img, kps2, None, 
                                                   color=(0, 255, 0), 
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Resize images for better display on dual monitors
            display_size = (640, 480)  # Smaller size to fit multiple windows
            
            img1_resized = cv2.resize(img1_with_keypoints, display_size)
            img2_resized = cv2.resize(img2_with_keypoints, display_size)
            
            # Zeige beide Bilder mit Keypoints als Kreise
            cv2.imshow("Source Frame Keypoints", img1_resized)
            cv2.imshow("Reference Image Keypoints", img2_resized)
            
            # Optional: Zeige auch die Matches mit Linien
            matches_img = cv2.drawMatches(
                frame, kps1,           # Source frame first (queryIdx)
                self.ref_img, kps2,    # Reference image second (trainIdx)
                good_matches, None, 
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            # Resize matches image (it's wider, so different aspect ratio)
            matches_resized = cv2.resize(matches_img, (1600, 800))
            cv2.imshow("Feature Matches", matches_resized)

        h, w = self.ref_img.shape[:2]
        warped = cv2.warpPerspective(frame, self.H, (w, h))
        return True, warped


    def warp_frame(self, frame):
        if frame is None:
            raise ValueError("Frame cannot be None")

        h, w = self.ref_img.shape[:2]
        warped = cv2.warpPerspective(frame, self.H, (w, h))
        
        return warped
