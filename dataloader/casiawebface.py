import struct, torch
from pathlib import Path
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset


class CasiaRecordIODataset(Dataset):
    """
    CASIA-WebFace loader that works with either
      • binary MXNet train.idx  (8-byte header + 16-byte pairs)
      • plain-text  train.idx   (idx \t offset per line)
    """

    def __init__(self, folder, transform=None):
        folder = Path(folder)
        self.rec_path = folder / "train.rec"
        self.idx_path = folder / "train.idx"
        lst_path      = folder / "train.lst"
        self.transform = transform

        # ---------- 1. idx → label map (robust) ----------------------------
        self.idx2label = {}
        with open(lst_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 3:
                    continue
                if p[1].isdigit():                 # idx label path
                    rec_idx, identity = int(p[0]), int(p[1])
                elif p[-1].isdigit():             # idx path label
                    rec_idx, identity = int(p[0]), int(p[-1])
                else:                             # fallback: folder name
                    rec_idx = int(p[0])
                    identity = int(Path(p[1]).parts[-2])
                self.idx2label[rec_idx] = identity

        # ---------- 2. read train.idx (binary *or* text) -------------------
        rec_size = self.rec_path.stat().st_size
        pairs = []                                # (idx, offset)

        with open(self.idx_path, "rb") as f:
            first8 = f.read(8)                    # peek
            f.seek(0)
            is_text = all(32 <= b < 127 for b in first8)  # ASCII?

            if is_text:
                # plain-text mode
                for line in f:
                    if not line.strip():
                        continue
                    idx_str, pos_str, *_ = line.decode().split()
                    idx, pos = int(idx_str), int(pos_str)
                    if 0 <= pos < rec_size:
                        pairs.append((idx, pos))
            else:
                # binary MXNet mode
                f.read(8)                         # skip header
                while True:
                    buf = f.read(16)
                    if len(buf) < 16:
                        break
                    idx, pos = struct.unpack("<qq", buf)
                    if idx < 0 or pos < 0 or pos >= rec_size:
                        continue                 # drop sentinel
                    pairs.append((idx, pos))

        if not pairs:
            raise RuntimeError("train.idx contained no valid pairs!")

        pairs.sort(key=lambda kp: kp[0])          # order by idx
        self.record_keys = [k for k, _ in pairs]
        self.pos_of      = {k: p for k, p in pairs}

    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self.record_keys)

    def __getitem__(self, idx):
        rec_idx = self.record_keys[idx]
        pos     = self.pos_of[rec_idx]

        with open(self.rec_path, "rb") as f:
            f.seek(pos)
            length, _ = struct.unpack("<II", f.read(8))
            img_bytes = f.read(length - 8)

        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.idx2label[rec_idx], dtype=torch.long)
        return img, label