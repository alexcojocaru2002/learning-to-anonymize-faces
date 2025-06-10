import torch.nn.functional as F

from alignment import align
from models.sphereface import AngleLoss


def adversarial_loss(modifier, classifier, faces, identity_labels, mode='M'):
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
    modified_faces = modifier(aligned_f)

    # Get classifier outputs: (cos_theta, phi_theta) from AngleLinear

    output_real = classifier(aligned_f)
    output_fake = classifier(modified_faces)

    criterion = AngleLoss()

    if mode == 'M':
        # Minimize classification on modified faces only (fool D)
        loss = criterion(output_fake, identity_labels)
    elif mode == 'D':
        # Maximize classification on modified faces and real faces → we minimize negative loss
        loss_real = criterion(output_real, identity_labels)
        loss_fake = criterion(output_fake, identity_labels)
        loss = loss_real + loss_fake
    else:
        raise ValueError("Mode must be 'M' or 'D'")

    return loss