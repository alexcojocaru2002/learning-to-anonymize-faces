from collections import defaultdict
from random import random

from torchvision.transforms import transforms

from alignment import align
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloader.casiawebface import CasiaRecordIODataset
from dataloader.jhmdb import JHMDBFrameDetDataset
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

        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")

def train_validation_split(vtransform, ftransform):
    vtrain = JHMDBFrameDetDataset("data/JHMDB", split="train", transform=vtransform)
    vval = JHMDBFrameDetDataset("data/JHMDB", split="test", transform=vtransform)
    fdataset = CasiaRecordIODataset("data/faces_webface_112x112", transform=ftransform)

    def collate_fn(batch):
        return tuple(zip(*batch))

    vtrain_loader = DataLoader(vtrain, batch_size=4, shuffle=True, collate_fn=collate_fn)
    vval_loader = DataLoader(vval, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # -------------- group indices by identity ------------------------------
    id_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(fdataset):
        id_to_indices[int(label)].append(idx)

    identity_ids = list(id_to_indices.keys())
    random.seed(42)  # reproducible
    random.shuffle(identity_ids)

    # 80 % identities → train, 20 % → val
    split_point = int(0.8 * len(identity_ids))
    train_ids, val_ids = identity_ids[:split_point], identity_ids[split_point:]

    train_idx = [i for pid in train_ids for i in id_to_indices[pid]]
    val_idx = [i for pid in val_ids for i in id_to_indices[pid]]

    ftrain = Subset(fdataset, train_idx)
    fval = Subset(fdataset, val_idx)

    ftrain_loader = DataLoader(ftrain, batch_size=64, shuffle=True, num_workers=4)
    fval_loader = DataLoader(fval, batch_size=64, shuffle=False, num_workers=4)

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
        T.ToTensor()
    ])

    vtrain_loader, vval_loader, ftrain_loader, fval_loader = train_validation_split(vtransform, ftransform)

    train(model, vtrain_loader, ftrain_loader, 12, 10, lambda_weight=1, optimizer_m=optimizer_m, optimizer_d=optimizer_d, optimizer_a=optimizer_a)
