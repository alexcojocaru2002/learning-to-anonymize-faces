from PIL import Image
from matplotlib import transforms
from mtcnn import MTCNN
import numpy as np
import cv2
import torch

def align(image, bbox=None, keypoints=None, output_size=(112, 96)):
    orig_is_tensor = isinstance(image, torch.Tensor)
    if isinstance(image, torch.Tensor):
        image = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    else:
        image = np.clip(image, 0, 1)  
            
    if keypoints is not None and len(keypoints) == 5:
        template = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)
        template[:, 0] += 8.0  
        template[:, 0] *= output_size[1] / 96.0  
        template[:, 1] *= output_size[0] / 112.0 

        keypoints = np.array(keypoints, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(keypoints, template)
        aligned = cv2.warpAffine(image, M, output_size, borderValue=0.0)
        
        if orig_is_tensor:
            if aligned.dtype == np.uint8:
                aligned = aligned.astype(np.float32) / 255.0
            else:
                aligned = aligned.astype(np.float32)
            aligned = torch.from_numpy(aligned).permute(2, 0, 1).contiguous()
        return aligned

    elif bbox is not None:
        x1,y1,x2,y2 = bbox
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, output_size)
      
        if resized.dtype == np.uint8:
            resized = resized.astype(np.float32) / 255.0
        else:
            resized = resized.astype(np.float32)
        resized = torch.from_numpy(resized).permute(2, 0, 1).contiguous()
                
        return resized

    else:
        raise ValueError("Either keypoints or bbox must be provided.")
    
# def align_batch(images, bboxes=None, keypoints_list=None, output_size=(112, 112)):
#     aligned_faces = []
#     for idx, image in enumerate(images):
#         img = cv2.imread(filepath)
#         mtcnn = MTCNN()
#         detections = mtcnn.detect_faces(img)
#         kp = detections[0].get('keypoints')
#         keypoints = [
#             kp['left_eye'],
#             kp['right_eye'],
#             kp['nose'],
#             kp['mouth_left'],
#             kp['mouth_right']
#         ]
#         bbox = detections[0].get('box')
#         aligned = align(image, bbox=bbox, keypoints=keypoints, output_size=output_size)
#         aligned_faces.append(aligned)
    
#     return aligned_faces