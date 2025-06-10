from collections import defaultdict
from random import random

from torchvision.transforms import transforms

from alignment import align
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloader.casiawebface import CasiaDataset
from dataloader.jhmdb import JHMDBFrameDetDataset
from losses.adversarial_loss import adversarial_loss
from models.our_model import OurModel
import torchvision.transforms as T
import torch.nn.functional as F


def train(model, loader_video, loader_faces, T1, T2, lambda_weight, optimizer_d, optimizer_a, optimizer_m):
    video_iter = iter(loader_video)
    faces_iter = iter(loader_faces)
    for epoch in tqdm(range(T1)):

        # 2 for clip_tuple, 4 for face_tuple
        for clip_tuple, face_tuple in zip(video_iter, faces_iter):

            print(clip_tuple[0].shape)
            print(face_tuple[0].shape)

            v = clip_tuple[0]
            vlabels = clip_tuple[1]
            f = face_tuple[0]
            f_n = face_tuple[1]
            flabels = face_tuple[2]
            flabels_n = face_tuple[3]


            # print(f.shape)
            # print(aligned_f.shape)
            l_adv = adversarial_loss(model.m, model.d, f, flabels, mode='D')
            # argmax update on D
            optimizer_d.zero_grad()
            l_adv.backward()
            optimizer_d.step()

            # input video frame v
            print("DONE ONE BACKWARDS PASS")
            bounding_boxes = model.face_detection.detect_faces_yolo(v)
            l1 = 0
            if len(bounding_boxes) > 0:
                r_v, v_prime = model.face_detection.cut_regions(v, bounding_boxes)

                # Expects r_v to be (B, 3, H, W)
                rv_prime = model.m(r_v)
                v_prime = v_prime + rv_prime  # have to double check if adding images works like this
                l1 = F.l1_loss(rv_prime, r_v)
            else:
                v_prime = v

            l_det_dict = model.a(v_prime, vlabels)
            l_det = sum(l_det_dict.values()) # sum of the loss values

            # argmin M, A update
            l_adv = adversarial_loss(model.m, model.d, f, flabels, mode='M')
            final_loss = l_adv + l_det + lambda_weight * l1
            optimizer_a.zero_grad()
            optimizer_m.zero_grad()
            final_loss.backward()
            optimizer_a.step()
            optimizer_m.step()

        #second for for fine tuning A

        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")

def train_validation_split(vtransform, ftransform):
    vtrain = JHMDBFrameDetDataset("data/JHMDB", split="train", transform=vtransform)
    vval = JHMDBFrameDetDataset("data/JHMDB", split="test", transform=vtransform)

    vtrain_loader = DataLoader(vtrain, batch_size=20, shuffle=True)
    vval_loader = DataLoader(vval, batch_size=20, shuffle=False)
    print("Loaded JHMDB")
    fdataset = CasiaDataset("data/casia_webface_images", transform=ftransform)

    # -------------- group indices by identity ------------------------------
    # 60 % → train, 20 % → val, 20 % → test

    ftrain, fval, ftest = torch.utils.data.random_split(fdataset, [0.6, 0.2, 0.2])

    ftrain_loader = DataLoader(ftrain, batch_size=20, shuffle=True, num_workers=4)
    fval_loader = DataLoader(fval, batch_size=20, shuffle=False, num_workers=4)
    ftest_loader = DataLoader(ftrain, batch_size=20, shuffle=False, num_workers=4)

    print("Loaded Casia Webface")

    return vtrain_loader, vval_loader, ftrain_loader, fval_loader

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
    vtransform = T.Compose([
        T.ToTensor()
    ])

    ftransform = T.Compose([
        T.ToTensor(),

    ])

    vtrain_loader, vval_loader, ftrain_loader, fval_loader = train_validation_split(vtransform, ftransform)

    print("Starting training!")
    train(model, vtrain_loader, ftrain_loader, 12, 10, lambda_weight=1, optimizer_m=optimizer_m, optimizer_d=optimizer_d, optimizer_a=optimizer_a)
