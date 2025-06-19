import torch
import torch.nn.functional as F
from torch import nn

from alignment import align
from models.sphereface import AngleLoss
from scripts import utils


def adversarial_loss(modifier, classifier, faces, identity_labels, batch, device='cpu', mode='M', lambda_weight=7.5):
    """
    Computes adversarial loss between modifier M and identity classifier D.

    Args:
        modifier (nn.Module): The face modifier model (M).
        classifier (nn.Module): The identity classifier with angular softmax head (D).
        faces (Tensor): Input face images of shape (B, C, H, W).
        identity_labels (Tensor): Corresponding identity labels (B,).
        mode (str): Either 'M' for updating the modifier or 'D' for updating the classifier.

    Returns:
        loss (Tensor): Scalar loss value.
    """
    # Generate modified faces
    aligned_f = align.align_batch(faces)
    aligned_f = aligned_f.to(device)

    aligned_f = aligned_f * 2 - 1 # have to scale to 1, 1 from 0, 1 since we use tanh as last layer
    aligned_f_resized = F.interpolate(aligned_f, size=(256, 256), mode='bilinear', align_corners=False) #detail from the paper, section 4, face modification

    modified_faces = modifier(aligned_f_resized)
    modified_faces = F.interpolate(modified_faces, size=(112, 96), mode='bilinear', align_corners=False)

    if batch % 50 == 0 and mode == 'M':
        modified_faces_vis = modified_faces
        aligned_f_vis = aligned_f
        modified_faces_vis = (modified_faces_vis + 1) / 2
        aligned_f_vis = (aligned_f_vis + 1) / 2 # scale back to 0,1 for visualization
        utils.show_images(torch.cat((modified_faces_vis, aligned_f_vis), dim=0))

    # Get classifier outputs: (cos_theta, phi_theta) from AngleLinear

    output_real = classifier(aligned_f) # also expects -1, 1 as input
    output_fake = classifier(modified_faces)

    predicted_idx = output_real[0].argmax(dim=1)    # class index
    predicted_idx_fake = output_fake[0].argmax(dim=1)

    # print(predicted_idx)
    # print(predicted_idx_fake)
    # print(identity_labels)
    criterion = AngleLoss()
    l1_loss = 0

    if mode == 'M':
        # Minimize classification on modified faces only (fool D)
        loss = -criterion(output_fake, identity_labels)
        if loss > 100:
            print("HERE")
        # Add l1 loss to preserve face structure brightness... etc
        l1 = torch.abs(aligned_f - modified_faces)  # [B, C, H, W]
        l1_per_image = l1.view(l1.size(0), -1).mean(dim=1)  # [B] → mean over
        l1_loss = lambda_weight * l1_per_image.mean()

    elif mode == 'D':
        # Maximize classification on modified faces and real faces → we minimize negative loss
        loss_real = criterion(output_real, identity_labels)
        loss_fake = criterion(output_fake, identity_labels)
        loss = loss_real + loss_fake
        if loss > 100:
            print("HERE")
    else:
        raise ValueError("Mode must be 'M' or 'D'")

    return loss, l1_loss