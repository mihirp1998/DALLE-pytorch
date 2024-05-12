import glob
import json
import os
import os.path as osp
import torch
import numpy as np
import pandas as pd
from torchvision import datasets
from PIL import Image
import matplotlib.pyplot as plt

class NYUDepth(torch.utils.data.Dataset):
    def __init__(self, images_root, split, transform=lambda x: x):
        super().__init__()
        self._images_root = images_root

        df = pd.read_csv(osp.join(self._images_root, "data", f"{split}.csv"), header=None)
        df[0] = df[0].apply(lambda x: osp.join(self._images_root, x))
        df[1] = df[1].apply(lambda x: osp.join(self._images_root, x))

        print("Samples ", len(df))

        self._images = df[0].values.tolist()
        self._targets = df[1].values.tolist()
        self.transform = transform


    def colored_depthmap(self, depth, d_min=None, d_max=None, cmap=plt.cm.inferno):
        if d_min is None:
            d_min = np.min(depth)
        if d_max is None:
            d_max = np.max(depth)
        depth_relative = (depth - d_min) / (d_max - d_min)
        
        return (255 * cmap(depth_relative)[:, :, :3]).astype(np.uint8)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        
        image_filepath = self._images[index]
        dmap_filepath = self._targets[index]

        img, depth_target = Image.open(image_filepath).convert("RGB"), Image.open(dmap_filepath).convert("L")
        depth_target = np.squeeze(np.array(depth_target))

        d_min = np.min(depth_target)
        d_max = np.max(depth_target)
        depth_target = self.colored_depthmap(depth_target, d_min, d_max)
        depth_target = Image.fromarray(depth_target)

        if self.transform is not None:
            img = self.transform(img)
            depth_target = self.transform(depth_target)
        
        return img, depth_target