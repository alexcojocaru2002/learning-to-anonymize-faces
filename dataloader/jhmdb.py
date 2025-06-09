import os, pickle, math
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image


class JHMDBFrameDetDataset(Dataset):
    """
    Per-frame action-detection dataset for *jhmdbb-gt.pkl* (gttubes version).

    Each __getitem__ returns:
        img     : torch.FloatTensor  (C,H,W)
        target  : {
                    boxes     : FloatTensor (N,4)  N=1 here
                    labels    : LongTensor  (N,)
                    video_id  : str         "brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0"
                    frame_idx : LongTensor  scalar
                  }
    """

    def __init__(
        self,
        root: str,
        split: str = "train",    # 'train' | 'test'
        split_id: int = 0,       # which official split (0,1,2)
        transform=None,
    ):
        root        = Path(root)
        self.transform = transform
        pkl_path    = root / "JHMDB-GT.pkl"
        frames_root = root / "Frames"

        with open(pkl_path, "rb") as f:
            gt = pickle.load(f, encoding="latin1")

        self.labels = gt["labels"]                          # action names
        vlist       = gt[f"{split}_videos"][split_id]       # chosen videos
        gttubes     = gt["gttubes"]                         # video → dict[tube] → ndarray(T,5)

        self.samples: List[Dict[str, Any]] = []
        for video in vlist:
            class_name, video_name = video.split("/", 1)
            video_dir = frames_root / class_name / video_name

            for label_id, tube_list in gttubes[video].items():  # dict → list
                for tube in tube_list:  # ndarray (T,5)
                    for frame in tube:  # frame = (≥5,)
                        fnum, x1, y1, x2, y2 = frame[:5]
                        fnum = int(fnum.item())  # numpy scalar → int

                        fname = f"{fnum:05d}.png"
                        frame_path = video_dir / fname
                        if not frame_path.exists():
                            frame_path = video_dir / f"{fnum:03d}.png"
                        if not frame_path.exists():
                            continue

                        self.samples.append(
                            dict(
                                frame_path=frame_path,
                                boxes=[[float(x1), float(y1), float(x2), float(y2)]],
                                labels=[int(label_id)],
                                video_id=video,
                                frame_idx=fnum,
                            )
                        )

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["frame_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        target = {
            "boxes"    : torch.tensor(s["boxes"],  dtype=torch.float32),
            "labels"   : torch.tensor(s["labels"], dtype=torch.int64),
            "video_id" : s["video_id"],
            "frame_idx": torch.tensor(s["frame_idx"], dtype=torch.int64),
        }
        return img, target
