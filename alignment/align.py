import numpy as np
import cv2

def align(image, bbox=None, keypoints=None):
    if keypoints is not None:
        standard_positions = np.array([
          [30.2946, 51.6963],
          [65.5318, 51.5014],
          [48.0252, 71.7366],
          [33.5493, 92.3655],
          [62.7299, 92.2041] ], dtype=np.float32 )
        standard_positions[:,0] += 8.0
        dst = keypoints.astype(np.float32)
        M = cv2.estimateAffine2D(dst,standard_positions)[0]
        warped = cv2.warpAffine(image,M,(112,112), borderValue = 0.0)
        return warped
    else:
        x1,y1,x2,y2,_ = bbox
        ret = image[y1:y2,x1:x2]
        ret = cv2.resize(ret, (112,112))
        return ret