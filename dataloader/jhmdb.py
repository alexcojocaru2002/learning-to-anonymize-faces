import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class JHMDBFramesDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, transform=None):
        self.root_dir = root_dir  # e.g., "data/jhmdb/Frames"
        self.clip_len = clip_len
        self.transform = transform
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for class_name in sorted(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir) or class_name.startswith('.'):
                continue

            for video_name in sorted(os.listdir(class_dir)):
                video_dir = os.path.join(class_dir, video_name)
                if not os.path.isdir(video_dir) or video_name.startswith('.'):
                    continue

                frames = sorted(os.listdir(video_dir))
                frames = [f for f in frames if not f.startswith('.') and f.endswith(('.jpg', '.png'))]

                if len(frames) >= self.clip_len:
                    samples.append((video_dir, class_name, frames))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, class_name, frames = self.samples[idx]
        frame_indices = range(0, self.clip_len)
        clip = []
        for i in frame_indices:
            frame_path = os.path.join(video_dir, frames[i])
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            clip.append(image)

        import torch
        clip = torch.stack(clip, dim=0)  # Shape: (T, C, H, W)
        label = self._class_to_index(class_name)
        return clip, label

    def _class_to_index(self, class_name):
        class_names = sorted({s[1] for s in self.samples})
        return class_names.index(class_name)
