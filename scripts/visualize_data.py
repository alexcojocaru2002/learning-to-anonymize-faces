import os
import sys

from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image

from dataloader.jhmdb import JHMDBFramesDataset


def visualize_data():
    print("Showing image")

def run():
    transform = T.Compose([
        T.ToTensor()
    ])

    dataset = JHMDBFramesDataset("data/JHMDB/Frames", transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Get one batch
    data_iter = iter(loader)
    clips, labels = next(data_iter)

    # Save each frame
    save_dir = "data/JHMBD_saved_frames/"
    os.makedirs(save_dir, exist_ok=True)

    # Select first clip in the batch
    for j in range(0, clips.shape[0]):
        clip = clips[j]  # Shape: (T, C, H, W)
        label = labels[j]

        print(j)
        os.makedirs(save_dir + f"video_{j:03d}", exist_ok=True)

        for i, frame in enumerate(clip):
            img = to_pil_image(frame)
            img.save(os.path.join(save_dir + f"video_{j:03d}/" + f"frame_{i:03d}.png"))