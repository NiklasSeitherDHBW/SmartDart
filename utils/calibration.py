import cv2
import numpy as np


class CameraCalibration:
    def __init__(self, ref_img, debug=False):
        self.debug = debug
        if ref_img is not None:
            self.ref_img = cv2.imread(ref_img)
            if self.ref_img is None:
                raise ValueError(f"Could not load reference image: {ref_img}")

        self.orb = cv2.ORB_create(1000)
        self.kps_ref, self.des_ref = self.orb.detectAndCompute(self.ref_img, None)
        
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def warp_frame(self, frame):
        if frame is None:
            raise ValueError("Frame cannot be None")

        kps_frame, des_frame = self.orb.detectAndCompute(frame, None)
        if des_frame is None or self.des_ref is None:
            return None, None

        matches = self.bf.match(des_frame, self.des_ref)
        matches = sorted(matches, key=lambda x: x.distance)[:80]

        pts_src = np.float32([kps_frame[m.queryIdx].pt for m in matches])
        pts_ref = np.float32([self.kps_ref[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 5.0)
        
        h, w = self.ref_img.shape[:2]
        warped = cv2.warpPerspective(frame, H, (w, h))
        
        return warped
