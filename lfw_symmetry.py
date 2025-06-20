#!/usr/bin/env python

import os
import cv2
import kagglehub
import shutil

print("Downloading LFW dataset from Kaggle…")
dataset_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
print(f"Download complete. Dataset directory: {dataset_path}\n")

candidate_dirs = [
    os.path.join(dataset_path, d)
    for d in ("lfw-deepfunneled", "lfw", "lfw_funneled")
    if os.path.isdir(os.path.join(dataset_path, d))
]
LFW_ROOT = candidate_dirs[0] if candidate_dirs else dataset_path
print(f"Using LFW images from: {LFW_ROOT}\n")

OUT_ROOT = "data/lfw_resolution_test"
N_IMAGES = 30

if os.path.exists(OUT_ROOT):
    shutil.rmtree(OUT_ROOT)
os.makedirs(OUT_ROOT, exist_ok=True)

all_imgs = []
for root, _, files in os.walk(LFW_ROOT):
    for f in sorted(files):
        if f.lower().endswith(".jpg"):
            all_imgs.append(os.path.join(root, f))
        if len(all_imgs) >= N_IMAGES:
            break
    if len(all_imgs) >= N_IMAGES:
        break

print(f"Selected {len(all_imgs)} images for processing.\n")

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def downsample_then_up(img, target):
    small = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (250, 250), interpolation=cv2.INTER_CUBIC)

def gaussian_blur(img, ksize=9, sigma=0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

TRANSFORMS = [
    ("ds128", lambda im: downsample_then_up(im, 128)),
    ("ds64",  lambda im: downsample_then_up(im, 64)),
    ("ds32",  lambda im: downsample_then_up(im, 32)),
    ("blur",   lambda im: gaussian_blur(im, 9, 0)),   # σ≈1.7 automatically chosen by OpenCV
    ("blur3",  lambda im: gaussian_blur(im, 9, 3)),   # explicit σ≈3
]

for src_path in all_imgs:
    img = cv2.imread(src_path)
    if img is None:
        print(f"⚠️  Skipping unreadable file: {src_path}")
        continue

    rel = os.path.relpath(src_path, LFW_ROOT)
    dst_orig = os.path.join(OUT_ROOT, rel)
    _ensure_dir(dst_orig)
    cv2.imwrite(dst_orig, img)

    for tag, fn in TRANSFORMS:
        variant = fn(img)
        name, ext = os.path.splitext(rel)
        dst_path = os.path.join(OUT_ROOT, f"{name}_{tag}{ext}")
        _ensure_dir(dst_path)
        cv2.imwrite(dst_path, variant)
        print(f"✅ {tag:6} -> {dst_path}")

print("\nDone! Check the `lfw_resolution_test/` folder for output images.")
