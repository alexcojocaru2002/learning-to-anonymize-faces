from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from matplotlib import transforms
import numpy as np
import cv2
import torch

def align(image, bbox=None, keypoints=None, output_size=(112, 96)):
    orig_is_tensor = isinstance(image, torch.Tensor)
    if orig_is_tensor:
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
        aligned = cv2.warpAffine(image, M, (output_size[1], output_size[0]), borderValue=0.0)
        
        if aligned.dtype == np.uint8:
            aligned = aligned.astype(np.float32) / 255.0
        else:
            aligned = aligned.astype(np.float32)
        if aligned.ndim == 2:
            aligned = np.expand_dims(aligned, axis=-1)
        if aligned.shape[-1] == 1:
            aligned = np.repeat(aligned, 3, axis=-1)
        aligned = torch.from_numpy(aligned).permute(2, 0, 1).contiguous()  # (C, H, W)
        return aligned

    elif bbox is not None:
        x1, y1, x2, y2 = bbox
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (output_size[1], output_size[0]))
        if resized.dtype == np.uint8:
            resized = resized.astype(np.float32) / 255.0
        else:
            resized = resized.astype(np.float32)
        if resized.ndim == 2:
            resized = np.expand_dims(resized, axis=-1)
        if resized.shape[-1] == 1:
            resized = np.repeat(resized, 3, axis=-1)
        resized = torch.from_numpy(resized).permute(2, 0, 1).contiguous()
        return resized

    else:
        raise ValueError("Either keypoints or bbox must be provided.")
    
def align_batch(images):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=False, device=device)
    img_list = [
        (np.transpose(img.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
        for img in images
    ]
    aligned_faces = mtcnn(img_list)
    for i, face in enumerate(aligned_faces):
        if face is None:
            aligned_faces[i] = images[i]  
    return torch.stack(aligned_faces)