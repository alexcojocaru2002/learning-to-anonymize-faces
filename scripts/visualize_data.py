import os
import sys

from PIL import Image
import cv2
from facenet_pytorch.models.mtcnn import MTCNN
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import ToPILImage, transforms
from torchvision.transforms.functional import to_pil_image

from alignment.align import align
from dataloader.jhmdb import JHMDBFrameDetDataset
from models.face_recognition import MyFaceIdYOLOv8


def load_image_as_tensor(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to (C, H, W) and scales to [0, 1]
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    return tensor  # Shape: (C, H, W)

def visualize_data():
    print("Showing image")

def run():
    transform = T.Compose([
        T.ToTensor()
    ])

    dataset = JHMDBFrameDetDataset("data/JHMDB/Frames", transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize face recognizer
    face_model = MyFaceIdYOLOv8()

    # Get one batch
    data_iter = iter(loader)
    clips, labels = next(data_iter)

    # Directory for saving frames (optional)
    save_dir = "data/JHMBD_saved_frames/"
    os.makedirs(save_dir, exist_ok=True)

    # Use the first clip in the batch
    clip = clips[0]  # Shape: (T, C, H, W)
    label = labels[0]

    print("Running face detection on first frame of the clip...")
    first_frame = clip[0]  # tensor of shape (C, H, W)

    # Run face recognition on this frame
    face_model.visualize(first_frame)

    # img = load_image_as_tensor('data/domnu.png')
    # img_np = np.transpose(img.numpy(), (1, 2, 0))
    # img_np_uint8 = (img_np * 255).astype(np.uint8)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # mtcnn = MTCNN(keep_all=True, device=device)
    # boxes, probs, landmarks = mtcnn.detect(img_np_uint8, landmarks=True)
    # print("Boxes:", boxes)
    # print("Landmarks:", landmarks)
    # img_aligned = align(img, boxes[0].astype(int), landmarks[0])
    
    # print("Aligned image shape:", img_aligned.shape)
    # print(isinstance(img_aligned, torch.Tensor))
    # if isinstance(img_aligned, torch.Tensor):
    #     img_vis = img_aligned.detach().cpu().numpy()
    #     if img_vis.shape[0] == 3:  # (C, H, W)
    #         img_vis = np.transpose(img_vis, (1, 2, 0))
    #     img_vis = np.clip(img_vis, 0, 1)  # If in [0,1]
    #     plt.imshow(img_vis)
    #     plt.title("Aligned Image")
    #     plt.axis('off')
    #     plt.show()
    # else:
    #     print("img_aligned is not a tensor, cannot visualize directly.")
    
    # (Optional) Save all frames from the clip
    for i, frame in enumerate(clip):
        img = face_model.visualize(frame)
        video_dir = os.path.join(save_dir, f"video_000")
        os.makedirs(video_dir, exist_ok=True)