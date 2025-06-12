import glob
import os
import struct, torch
from pathlib import Path
from io import BytesIO
from numpy import random
from torch.utils.data import Dataset

from PIL import Image
from torch.utils.data import Dataset


class CasiaDataset(Dataset):
    def __init__(self, data_dir='data', transform=None):
        super().__init__()

        # make labels
        img_folder_name = os.listdir(data_dir)
        img_folder_name.sort()

        self.labels = {}
        for index, label in enumerate(img_folder_name):
            self.labels[label] = index

        # make images path
        images = glob.glob(data_dir + '/**/*.jpg', recursive=True)

        # make indexlist
        self.indexlist = []
        for image in images:
            self.indexlist.append(image + ' ' + str(self.labels[os.path.normpath(image).split('/')[-2]]))

        random.shuffle(self.indexlist)

        # transform
        self.transform = transform

    def sample_negative(self, a_cls):
        while True:
            rand = random.randint(0, len(self.indexlist) - 1)
            name, cls = self.indexlist[rand].split()[0:2]
            if cls != a_cls:
                break
        return rand

    def load_img(self, index):
        info = self.indexlist[index].split()
        cls = int(info[1])

        img = Image.open(info[0]).convert('RGB')
        return img, cls

    def __getitem__(self, index):
        # Get the index of each image in the triplet
        a_name, a_cls = self.indexlist[index].split()
        n_index = self.sample_negative(a_cls)

        _a, a_cls = self.load_img(index)
        _n, n_cls = self.load_img(n_index)

        # transform images if required
        if self.transform:
            img_a = self.transform(_a)
            img_n = self.transform(_n)
        else:
            img_a = _a
            img_n = _n
        return img_a, img_n, a_cls, n_cls

    def __len__(self):
        return len(self.indexlist)