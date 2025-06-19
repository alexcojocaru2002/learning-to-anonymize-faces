import gc
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
        model.a.train()
        model.m.train()
        model.d.train()

        # 2 for clip_tuple, 4 for face_tuple
        batch = 0
        for clip_tuple, face_tuple in zip(video_iter, faces_iter):

            v = clip_tuple[0].to(model.device)

            vlabels = clip_tuple[1]
            f = face_tuple[0]
            f_n = face_tuple[1].to(model.device) # negative samples, might be useful for the loss
            flabels = face_tuple[2].to(model.device)
            flabels_n = face_tuple[3].to(model.device) # negative samples, might be useful for the loss


            # print(f.shape)
            # print(aligned_f.shape)
            l_adv_d, l1_d = adversarial_loss(model.m, model.d, f, flabels, batch=batch, device=model.device, mode='D', lambda_weight=lambda_weight)
            l_adv_d = l_adv_d + l1_d
            # argmax update on D
            optimizer_d.zero_grad()
            l_adv_d.backward()
            optimizer_d.step()

            # input video frame v
            bounding_boxes = model.face_detection.detect_faces_yolo(v)
            if len(bounding_boxes) > 0:
                r_v, _ = model.face_detection.cut_regions(v, bounding_boxes)

                # Collect original crop sizes directly from r_v now
                crop_sizes = []
                for face_crop in r_v:
                    _, h_i, w_i = face_crop.shape
                    crop_sizes.append((h_i, w_i))

                # Resize each crop to 256x256 before modifier
                rv_resized_list = []
                for face_crop in r_v:
                    resized_crop = F.interpolate(face_crop.unsqueeze(0), size=(256, 256), mode='bilinear',
                                                 align_corners=False)
                    rv_resized_list.append(resized_crop)

                rv_resized = torch.cat(rv_resized_list, dim=0)  # shape: (B, 3, 256, 256)

                # Apply modifier (remember to scale to [-1, 1])
                rv_resized = model.m(rv_resized * 2 - 1)

                # Resize back to original crop sizes
                rv_prime_list = []
                for i, (h_i, w_i) in enumerate(crop_sizes):
                    resized_back = F.interpolate(rv_resized[i:i + 1], size=(h_i, w_i), mode='bilinear',
                                                 align_corners=False)
                    rv_prime_list.append(resized_back[0])

                # Scale back to [0, 1]
                rv_prime_list = [(crop + 1) / 2 for crop in rv_prime_list]

                # Modify original frames
                v_prime_list = []
                for i, box in enumerate(bounding_boxes):
                    image_idx, x1, y1, x2, y2 = box.int().tolist()
                    frame = v[image_idx].clone()
                    mod_crop = rv_prime_list[i]
                    frame[:, y1:y2, x1:x2] = mod_crop
                    v_prime_list.append(frame)

                # Final tensor
                v_prime = torch.stack(v_prime_list, dim=0)
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


            # print(v_prime.shape)
            v_prime = utils.resize_batch_jhmdb(v_prime, device=model.device) # have to resize shorter side be 340 as stated in original paper
            # print(v_prime.shape)
            l_det_dict = model.a(v_prime, thechosen_vlables)

            l_det = sum(l_det_dict.values()) # sum of the loss values
            # argmin M, A update
            l_adv_m, l1_m = adversarial_loss(model.m, model.d, f, flabels, batch=batch, device=model.device, mode='M', lambda_weight=lambda_weight)
            l_adv_m = l_adv_m + l1_m
            final_loss = l_adv_m + l_det

            print(f"Final Loss is {final_loss.item()}, Adversarial loss for D is {l_adv_d} Adversarial loss for M is {l_adv_m} and Detection loss is {l_det.item()} L1 loss is {l1_m.item()}" )

            optimizer_a.zero_grad()
            optimizer_m.zero_grad()
            final_loss.backward()
            if batch % 50 == 0:
                torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")
                print(f"Gradient norms at batch {batch}:")
                for name, p in model.m.named_parameters():
                    if p.grad is not None:
                        grad_norm = p.grad.detach().abs().mean().item()
                        print(f"  {name}: {grad_norm:.6e}")
                    else:
                        print(f"  {name}: No gradient")
            optimizer_a.step()
            optimizer_m.step()
            if batch % 50 == 0:
                utils.show_images(torch.stack([v_prime[0]]), 2)
                utils.show_images(torch.stack([v[0]]), 2)

            batch += 1
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("Finished an epoch! Saving model...")
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

    vtrain_loader = DataLoader(vtrain, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
    # vval_loader = DataLoader(vval, batch_size=20, shuffle=False, collate_fn=custom_collate_fn)
    print("Loaded JHMDB")
    fdataset = CasiaDataset("data/casia_webface_images", transform=ftransform)

    # -------------- group indices by identity ------------------------------
    # 60 % → train, 20 % → val, 20 % → test

    ftrain, fval, ftest = torch.utils.data.random_split(fdataset, [0.6, 0.2, 0.2])

    ftrain_loader = DataLoader(ftrain, batch_size=2, shuffle=True, num_workers=4)
    # fval_loader = DataLoader(fval, batch_size=20, shuffle=False, num_workers=4)
    # ftest_loader = DataLoader(ftrain, batch_size=20, shuffle=False, num_workers=4)

    print("Loaded Casia Webface")

    return vtrain_loader, None, ftrain_loader, None

def run(num_output_classes=21):
    lambda_weights = [0.0, 0.01, 0.1, 1.0, 10.0]
    
    for lambda_weight in lambda_weights:
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

        train(model, vtrain_loader, ftrain_loader, 12, 10, lambda_weight=0.5, optimizer_m=optimizer_m, optimizer_d=optimizer_d, optimizer_a=optimizer_a)

def show_input_data(loader_video, loader_faces, device):
    N = 20
    video_iter = itertools.islice(iter(loader_video), N)
    faces_iter = itertools.islice(iter(loader_faces), N)

    for clip_tuple, face_tuple in zip(video_iter, faces_iter):
        v = clip_tuple[0].to(device)
        utils.show_images(v, 1)


if __name__ == "__main__":
    run()
