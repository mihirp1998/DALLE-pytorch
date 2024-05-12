import glob
import json
import os
import os.path as osp
import torch
from torchvision import datasets
from PIL import Image

class CocoSegmentation(torch.utils.data.Dataset):
    def __init__(self, images_root, split, transform=lambda x: x):
        super().__init__()
        self._images_root = images_root
        self._images = sorted(glob.glob(self._images_root +  f'/images/{split}/*'))
        self._masks = sorted(glob.glob(self._images_root + f'/annotations/semantic_{split}/*'))
        self.transform = transform

        print("Samples ", len(self._images), len(self._masks))

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image_filepath = self._images[index]
        mask_filepath = self._masks[index]

        img, mask = Image.open(image_filepath).convert("RGB"), Image.open(mask_filepath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask