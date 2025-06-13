import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F


def show_images(tensor, max_images=10):
    """
    Displays a batch of images stored in a tensor.

    Parameters:
    - tensor: torch.Tensor of shape (B, C, H, W) or (B, H, W) or (B, H, W, C)
    - max_images: maximum number of images to display
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach()
        if tensor.ndim == 3:  # (B, H, W) -> add channel dimension
            tensor = tensor.unsqueeze(1)

    # Handle numpy input as well
    elif isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    batch_size = tensor.shape[0]
    num_images = min(batch_size, max_images)

    plt.figure(figsize=(num_images * 2, 2))

    for i in range(num_images):
        img = tensor[i]

        if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
            img = img.permute(1, 2, 0)  # to (H, W, C)
        elif img.ndim == 3 and img.shape[-1] in [1, 3]:  # (H, W, C)
            pass
        elif img.ndim == 2:
            pass  # grayscale
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        plt.subplot(1, num_images, i + 1)
        if img.shape[-1] == 1:
            plt.imshow(img.squeeze(-1), cmap='gray')
        elif img.shape[-1] == 3:
            plt.imshow(img)
        else:  # grayscale
            plt.imshow(img, cmap='gray')

        plt.axis('off')

    plt.show()

# Used to resize jhmdb, mentioned in paper section 4 face modification
def resize_batch_jhmdb(batch_images, target_shorter_side=340, device='cuda'):
    """
    Resize a batch of images to have shorter side = 340 while preserving aspect ratio.
    Args:
        batch_images (Tensor): (B, 3, H, W), values in [0, 1].
        target_shorter_side (int): Target size for shorter side.
        device (str): torch device.
    Returns:
        List of resized tensors, each still in [0, 1].
    """
    batch_images = batch_images.to(device)
    resized_batch = []

    for img in batch_images:
        _, h, w = img.shape
        if h < w:
            scale = target_shorter_side / h
            new_h = target_shorter_side
            new_w = int(w * scale)
        else:
            scale = target_shorter_side / w
            new_w = target_shorter_side
            new_h = int(h * scale)

        resized_img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
        resized_batch.append(resized_img.squeeze(0))

    return torch.stack(resized_batch)


