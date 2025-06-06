import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import cv2

class MyFaceIdYOLOv8:
    def __init__(self, model_path='weights/yolov8n-face.pt'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect_faces_yolo(self, tensor_rgb):
        """
        Detect faces using YOLOv8 face model.
        Input: tensor_rgb (3, H, W) in range [0, 1]
        Output: list of dicts with 'bbox' and 'keypoints'
        """
        # Convert tensor to numpy image
        img_np = tensor_rgb.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        # YOLO expects BGR
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # Run inference
        results = self.model.predict(img_bgr, verbose=False)[0]
        detections = []
        for box in results.boxes:
            if float(box.conf[0]) > 0.5: # if confidence is high enough
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                keypoints = []
                if hasattr(box, "keypoints") and box.keypoints is not None:
                    keypoints = box.keypoints[0].cpu().int().tolist()
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "keypoints": keypoints  # Will be empty if not supported
                })

        return detections

    # v is frame from video as a tensor
    def cut_regions(self, v, bounding_boxes):
        crops = []
        v_copy = v.clone()  # avoid modifying original if needed

        for box in bounding_boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            crop = v_copy[:, y1:y2, x1:x2]
            crops.append(crop)  # list of cropped face regions

            # Zero out that region in the original tensor
            v_copy[:, y1:y2, x1:x2] = 0

        return crops, v_copy

    def visualize(self, tensor_rgb, figsize=(8, 6)):
            """
            Visualize detected faces and keypoints on the input image.
            """
            img_np = tensor_rgb.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            detections = self.detect_faces_yolo(tensor_rgb)

            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(img_np)
            ax.axis('off')

            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor='cyan', facecolor='none')
                ax.add_patch(rect)

                for kp in det['keypoints']:
                    ax.plot(kp[0], kp[1], 'ro', markersize=4)

            plt.tight_layout()
            plt.show()
