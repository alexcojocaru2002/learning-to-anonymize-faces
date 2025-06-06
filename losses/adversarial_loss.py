import torch.nn.functional as F

def adversarial_loss(modified_faces, faces, identity_labels, classifier):
    """
    modifier: M, a model that modifies faces
    classifier: D, the identity classifier
    faces: tensor (B, C, H, W)
    identity_labels: tensor (B,) of integer identity classes
    """
    # Pass through modifier

    # Classifier predictions
    logits_real = classifier(faces)
    logits_fake = classifier(modified_faces)

    # Cross-entropy losses
    loss_real = F.cross_entropy(logits_real, identity_labels)
    loss_fake = F.cross_entropy(logits_fake, identity_labels)

    # Total adversarial loss (maximize fake loss, minimize real loss)
    return -loss_fake + loss_real