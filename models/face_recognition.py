from typing import Tuple, List

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
        # ── torch → NumPy, RGB → BGR  (vectorised, no Python loops) ─────────
        batch_bgr = (
            tensor_rgb.detach().clamp_(0, 1)  # [0,1]
            .mul_(255).byte()  # → uint8 0-255
            .permute(0, 2, 3, 1)  # N,H,W,C
            .cpu().numpy()[..., ::-1]  # swap channels → BGR
        )

        # turn 4-D array into list[HWC] as Ultralytics expects
        img_list = [img for img in batch_bgr]  # length = N

        # ── YOLOv8 inference (batched) ──────────────────────────────────────
        results_batch = self.model.predict(
            img_list, verbose=False, conf=0.5
        )  # list[Results], len=N

        # ── parse detections ────────────────────────────────────────────────
        rows = []

        for img_idx, results in enumerate(results_batch):
            for box in results.boxes:
                if float(box.conf[0]) < 0.5:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].round().int().tolist()
                rows.append([img_idx, x1, y1, x2, y2])

        if not rows:
            # no detections at all – return an empty (0,5) tensor
            return torch.empty((0, 5), dtype=torch.int)

        return torch.tensor(rows, dtype=torch.int)

    def cut_regions(
            frames: torch.Tensor,
            det_tensor: torch.Tensor,
            img_idx: int | None = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extracts crops from `frame` using YOLO-style detections and zeros
        those regions in a copy of the frame.

        Parameters
        ----------
        frame : torch.Tensor           # shape (C, H, W)
            The video frame.
        det_tensor : torch.Tensor      # shape (N, 5) or (N, 4)
            Detection rows.  If shape is (N, 5) the first column is img_idx.
            Columns are [x1, y1, x2, y2] in pixel coords (inclusive x1/y1,
            exclusive x2/y2, as in PyTorch slicing).
        img_idx : int | None, optional
            If det_tensor has 5 columns, supply the index of this frame.
            If None and det_tensor has 5 columns, all rows are used.

        Returns
        -------
        crops : list[torch.Tensor]
            List of cropped regions (each shape (C, h, w)).
        frame_zeroed : torch.Tensor
            A clone of `frame` with crop regions set to 0.
        """
        if det_tensor.numel() == 0:
            # nothing to do – just return the untouched clone
            return [], frames.detach().clone()

        # Split coords and (optionally) filter by image index
        if det_tensor.size(1) == 5:
            if img_idx is not None:
                det_tensor = det_tensor[det_tensor[:, 0] == img_idx]
            boxes = det_tensor[:, 1:5]
        else:  # already just coords
            boxes = det_tensor

        crops: list[torch.Tensor] = []
        frame_zeroed = frames.detach().clone()

        # Ensure integer slicing
        boxes = boxes.int()

        C, H, W = frame_zeroed.shape
        for x1, y1, x2, y2 in boxes:
            # clamp to frame bounds (avoids IndexError on edge boxes)
            x1 = torch.clamp(x1, 0, W)
            x2 = torch.clamp(x2, 0, W)
            y1 = torch.clamp(y1, 0, H)
            y2 = torch.clamp(y2, 0, H)

            crop = frame_zeroed[:, y1:y2, x1:x2].clone()  # slice then clone
            crops.append(crop)

            # zero the region in-place
            frame_zeroed[:, y1:y2, x1:x2] = 0

        return crops, frame_zeroed

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
