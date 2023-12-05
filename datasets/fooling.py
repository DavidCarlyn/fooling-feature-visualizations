import os
from random import shuffle
from tqdm import tqdm

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def train_transforms():
    cur_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    return lambda x: (cur_transforms(x) * 255) - 117

def test_transforms():
    cur_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    return lambda x: (cur_transforms(x) * 255) - 117

def normalize(x):
    return T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x)

class FoolingDataset(Dataset):
    """
    0: Visualization Image
    1: Real Image
    """

    def __init__(self, vis_root, imgnet_root, split="train", transforms=None, small_test=False):
        super().__init__()
        assert split in ["train", "test"], f"{split} is not a valid options. Select either train or test"

        self.transforms = transforms

        imgnet_classes = set()
        imgnet_paths = []
        for root, dirs, paths in tqdm(os.walk(imgnet_root), desc="Loading ImageNet data"):
            for p in paths:
                imgnet_paths.append(os.path.join(root, p))
                imgnet_classes.add(root.split(os.path.sep)[-1])

        imgnet_classes = sorted(list(imgnet_classes))
        cls_to_idx = {imgnet_classes[i]: i for i in range(len(imgnet_classes))}

        vis_paths = []
        # ignore first 100 classes for training set
        split_point = 99
        if split == "train":
            imgnet_paths = list(filter(lambda x: cls_to_idx[x.split(os.path.sep)[-2]] > split_point, imgnet_paths))
        elif split == "test":
            imgnet_paths = list(filter(lambda x: cls_to_idx[x.split(os.path.sep)[-2]] <= split_point, imgnet_paths))
        
        for root, dirs, paths in tqdm(os.walk(vis_root), desc="Loading Visualization data"):
            for p in paths:
                cls_lbl = int(root.split(os.path.sep)[-1])
                if split == "train" and cls_lbl > split_point:
                    vis_paths.append(os.path.join(root, p))
                elif split == "test" and cls_lbl <= split_point:
                    vis_paths.append(os.path.join(root, p))


        imgnet_data = list(zip(imgnet_paths, np.ones(len(imgnet_paths)).astype(np.uint8).tolist()))
        visual_data = list(zip(vis_paths, np.zeros(len(vis_paths)).astype(np.uint8).tolist()))
        self.data = imgnet_data + visual_data
        
        self.num_imgnet = len(imgnet_paths)
        self.num_vis = len(vis_paths)

        self.class_weights = torch.tensor([1 - (self.num_vis / len(self.data)), 1 - (self.num_imgnet / len(self.data))])
        if small_test:
            shuffle(self.data)
            self.data = self.data[:100]
        #shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, lbl = self.data[idx]
        img = Image.open(path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)
            #if lbl == 1:
            #    img = normalize(img)

        return img, lbl