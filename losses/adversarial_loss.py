import torch
import torch.nn.functional as F
from torch import nn

from alignment import align
from models.sphereface import AngleLoss
from scripts import utils


def adversarial_loss(modifier, classifier, faces, identity_labels, batch, device='cpu', mode='M', lambda_weight=7.0):
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

    if batch % 10 == 0 and mode == 'M':
        modified_faces_vis = modified_faces[0:1]
        aligned_f_vis = aligned_f[0:1]
        modified_faces_vis = (modified_faces_vis + 1) / 2
        aligned_f_vis = (aligned_f_vis + 1) / 2 # scale back to 0,1 for visualization
        utils.show_images(torch.cat((modified_faces_vis, aligned_f_vis), dim=0))

    # Get classifier outputs: (cos_theta, phi_theta) from AngleLinear

    output_real = classifier(aligned_f) # also expects -1, 1 as input
    output_fake = classifier(modified_faces)

    criterion = AngleLoss()
    l1_loss = 0

    if mode == 'M':
        # Minimize classification on modified faces only (fool D)
        loss = criterion(output_fake, identity_labels)
        # Add l1 loss to preserve face structure brightness... etc
        l1 = nn.functional.l1_loss(aligned_f, modified_faces)
        l1_loss = lambda_weight * l1
    elif mode == 'D':
        # Maximize classification on modified faces and real faces → we minimize negative loss
        loss_real = criterion(output_real, identity_labels)
        loss_fake = criterion(output_fake, identity_labels)
        loss = loss_real + loss_fake
    else:
        raise ValueError("Mode must be 'M' or 'D'")

    return loss + l1_loss