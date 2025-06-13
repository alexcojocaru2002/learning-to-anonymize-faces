import itertools
import time
from collections import defaultdict
from random import random

import cv2
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from alignment import align
import torch
from torch.utils.data import DataLoader, Subset, default_collate
from tqdm import tqdm

from dataloader.casiawebface import CasiaDataset
from dataloader.jhmdb import JHMDBFrameDetDataset
from losses.adversarial_loss import adversarial_loss
from models.our_model import OurModel
import torchvision.transforms as T
import torch.nn.functional as F

from scripts import utils


def train(model, loader_video, loader_faces, T1, T2, lambda_weight, optimizer_d, optimizer_a, optimizer_m):
    video_iter = iter(loader_video)
    faces_iter = iter(loader_faces)

    print("Starting training!")
    for epoch in tqdm(range(T1)):

        # 2 for clip_tuple, 4 for face_tuple
        batch = 0
        for clip_tuple, face_tuple in zip(video_iter, faces_iter):

            v = clip_tuple[0].to(model.device)
            vlabels = clip_tuple[1]
            f = face_tuple[0]
            f_n = face_tuple[1].to(model.device)
            flabels = face_tuple[2].to(model.device)
            flabels_n = face_tuple[3].to(model.device)


            # print(f.shape)
            # print(aligned_f.shape)
            l_adv = adversarial_loss(model.m, model.d, f, flabels, batch=batch, device=model.device, mode='D')
            # argmax update on D
            optimizer_d.zero_grad()
            l_adv.backward()
            optimizer_d.step()

            # input video frame v
            bounding_boxes = model.face_detection.detect_faces_yolo(v)
            l1 = 0
            if len(bounding_boxes) > 0:
                r_v, v_prime = model.face_detection.cut_regions(v, bounding_boxes)
                # Expects r_v to be (B, 3, H, W)
                rv_prime = model.m(r_v)
                rv_prime = (rv_prime + 1) / 2 # have to rescale from -1, 1 to 0, 1 since we use tanh

                rv_prime_2, _ = model.face_detection.cut_regions_2(rv_prime, bounding_boxes)

                v_prime = v_prime + rv_prime_2  # have to double check if adding images works like this

                l1 = F.l1_loss(rv_prime_2, r_v)
            else:
                batch += 1
                print("No faces detected, skipping ...")
                v_prime = v
                continue

            thechosen_vlables = []
            for i in bounding_boxes[:, 0].tolist():
                vlabels[i]['boxes'] = vlabels[i]['boxes'].to(model.device)
                vlabels[i]['labels'] = vlabels[i]['labels'].to(model.device)
                vlabels[i]['frame_idx'] = vlabels[i]['frame_idx'].to(model.device)
                thechosen_vlables.append(vlabels[i])

            # start = time.time()
            l_det_dict = model.a(v_prime, thechosen_vlables)
            # end = time.time()
            # print(f"FRCNN Took {end - start:.4f} seconds")

            l_det = sum(l_det_dict.values()) # sum of the loss values
            # argmin M, A update
            l_adv = adversarial_loss(model.m, model.d, f, flabels, batch=batch, device=model.device, mode='M')
            final_loss = l_adv + l_det + lambda_weight * l1
            print(f"Final Loss is {final_loss.item()}, Adversarial loss for D is {l_adv} and Detection loss is {l_det.item()}" )
            optimizer_a.zero_grad()
            optimizer_m.zero_grad()
            final_loss.backward()
            optimizer_a.step()
            optimizer_m.step()

            if batch % 20 == 0:
                print("Done an iteration, saving model...")
                utils.show_images(torch.stack([v[bounding_boxes[0, 0]], v_prime[0]]), 2)
            batch += 1
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")

    # second for for fine tuning A

def train_validation_split(vtransform, ftransform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vtrain = JHMDBFrameDetDataset("data/JHMDB", split="train", transform=vtransform)
    # vval = JHMDBFrameDetDataset("data/JHMDB", split="test", transform=vtransform)

    def custom_collate_fn(batch):
        # Custom collate function to handle variable length sequences
        clip_1, clip_2 = zip(*batch)
        return default_collate(clip_1), list(clip_2)

    vtrain_loader = DataLoader(vtrain, batch_size=5, shuffle=True, collate_fn=custom_collate_fn)
    # vval_loader = DataLoader(vval, batch_size=20, shuffle=False, collate_fn=custom_collate_fn)
    print("Loaded JHMDB")
    fdataset = CasiaDataset("data/casia_webface_images", transform=ftransform)

    # -------------- group indices by identity ------------------------------
    # 60 % → train, 20 % → val, 20 % → test

    ftrain, fval, ftest = torch.utils.data.random_split(fdataset, [0.6, 0.2, 0.2])

    ftrain_loader = DataLoader(ftrain, batch_size=5, shuffle=True, num_workers=4)
    # fval_loader = DataLoader(fval, batch_size=20, shuffle=False, num_workers=4)
    # ftest_loader = DataLoader(ftrain, batch_size=20, shuffle=False, num_workers=4)

    print("Loaded Casia Webface")

    return vtrain_loader, None, ftrain_loader, None

def run(num_output_classes=21):
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

    vtrain_loader, _, ftrain_loader, _ = train_validation_split(vtransform, ftransform)

    # show_input_data(vtrain_loader, ftrain_loader, model.device)

    train(model, vtrain_loader, ftrain_loader, 12, 10, lambda_weight=1, optimizer_m=optimizer_m, optimizer_d=optimizer_d, optimizer_a=optimizer_a)

def show_input_data(loader_video, loader_faces, device):
    N = 20
    video_iter = itertools.islice(iter(loader_video), N)
    faces_iter = itertools.islice(iter(loader_faces), N)

    for clip_tuple, face_tuple in zip(video_iter, faces_iter):
        v = clip_tuple[0].to(device)
        utils.show_images(v, 1)


if __name__ == "__main__":
    run()
