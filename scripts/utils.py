import numpy as np
import torch
from matplotlib import pyplot as plt


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