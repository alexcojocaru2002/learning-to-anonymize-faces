from alignment import align
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.jhmdb import JHMDBFramesDataset
from losses.adversarial_loss import adversarial_loss
from models.our_model import OurModel
import torchvision.transforms as T
import torch.nn.functional as F


def train(model, loader_video, loader_faces, T1, T2, lambda_weight, optimizer_d, optimizer_a, optimizer_m):
    video_iter = iter(loader_video)
    faces_iter = iter(loader_faces)
    for epoch in tqdm(range(T1)):
        for (clips, labels), (faces, flabels) in zip(video_iter, faces_iter):
            v = clips
            f = faces
            m_f = model.m(f)
            # align m_f
            detections = model.face_detection.detect_faces_yolo(m_f)
            aligned_faces = []
            for index, face in enumerate(m_f):
                aligned_faces.append(align(face, detections[index]['bbox'], detections[index]['keypoints']))
            
            l_adv = adversarial_loss(m_f, f, flabels, model.d)
            # argmax update on D
            optimizer_d.zero_grad()
            optimizer_m.zero_grad()
            l_adv.backward()
            optimizer_d.step()
            optimizer_m.step()

            # input video frame v
            bounding_boxes = model.face_detection.detect_faces_yolo(v)
            l1 = 0
            if len(bounding_boxes) > 0:
                r_v, v_prime = model.cut_regions(v, bounding_boxes)

                # Expects r_v to be (B, 3, H, W)
                rv_prime = model.m(r_v)
                v_prime = v_prime + rv_prime  # have to double check if adding images works like this
                l1 = F.l1_loss(rv_prime, r_v)
            else:
                v_prime = v

            l_det_dict = model.a(v_prime, labels)
            l_det = sum(l_det_dict.values()) # sum of the loss values

            # argmin M, A update
            final_loss = l_adv + l_det + lambda_weight * l1
            optimizer_a.zero_grad()
            final_loss.backward()
            optimizer_a.step()

    #second for for fine tuning A


def run(num_output_classes=10):
    # Initialize model and learning rates
    model = OurModel(num_output_classes)
    lr_m = lr_d = 0.0003
    lr_a = 0.001
    optimizer_m = torch.optim.Adam(model.m.parameters(), betas=(0.5, 0.999), lr=lr_m)
    optimizer_d = torch.optim.Adam(model.d.parameters(), betas=(0.5, 0.999), lr=lr_d)
    optimizer_a = torch.optim.Adam(model.a.parameters(), betas=(0.5, 0.999), lr=lr_a)

    # Transformations for our dataloader
    # Storing images as tensors
    transform = T.Compose([
        T.ToTensor()
    ])

    dataset = JHMDBFramesDataset("data/JHMDB/Frames", transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)


    train(model, loader, loader, 12, 10, lambda_weight=1, optimizer_m=optimizer_m, optimizer_d=optimizer_d, optimizer_a=optimizer_a)