import os
import sys

from PIL import Image
import cv2
from mtcnn import MTCNN
import numpy as np
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

    # img = load_image_as_tensor('data/lucian.png')
    # detections = face_model.detect_faces_yolo(img)
    # img = cv2.imread('data/monica.png')
    # detector = MTCNN()
    # detections = detector.detect_faces(img)
    # kp = detections[0].get('keypoints')
    # keypoints = [
    #     kp['left_eye'],
    #     kp['right_eye'],
    #     kp['nose'],
    #     kp['mouth_left'],
    #     kp['mouth_right']
    # ]
    # img_aligned = align(img, detections[0].get('box'), keypoints)
    
    # (Optional) Save all frames from the clip
    for i, frame in enumerate(clip):
        img = face_model.visualize(frame)
        video_dir = os.path.join(save_dir, f"video_000")
        os.makedirs(video_dir, exist_ok=True)