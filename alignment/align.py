from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from matplotlib import transforms
import numpy as np
import cv2
import torch

def align(image, bbox=None, keypoints=None, output_size=(112, 96)):
    # This is essentially your original align() function
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
    else:
        # fallback: simple center-crop and resize
        h, w, _ = image.shape
        min_side = min(h, w)
        cx, cy = w // 2, h // 2
        half_side = min_side // 2
        cropped = image[cy-half_side:cy+half_side, cx-half_side:cx+half_side]
        aligned = cv2.resize(cropped, (output_size[1], output_size[0]))

    aligned = aligned.astype(np.float32)
    if aligned.ndim == 2:
        aligned = np.expand_dims(aligned, axis=-1)
    if aligned.shape[-1] == 1:
        aligned = np.repeat(aligned, 3, axis=-1)
    aligned = torch.from_numpy(aligned).permute(2, 0, 1).contiguous()
    return aligned

def align_batch(images):
    aligned_faces = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    for image in images:
        img_np = np.transpose(image.numpy(), (1, 2, 0))
        img_np_uint8 = (img_np * 255).astype(np.uint8)

        boxes, probs, landmarks = mtcnn.detect(img_np_uint8, landmarks=True)

        if boxes is not None and landmarks is not None:
            aligned_face = align(image, boxes[0].astype(int), landmarks[0])
        else:
            aligned_face = align(image)  # fallback case

        aligned_faces.append(aligned_face)

    return torch.stack(aligned_faces)

