import os

from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T

def train_transforms():
    return T.Compose([
        T.CenterCrop((256, 256)),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def test_transforms():
    return T.Compose([
        T.CenterCrop((256, 256)),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

class CUB_Dataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        super().__init__()

        self.transforms = transforms

        data_ids = []
        with open(os.path.join(root, "train_test_split.txt")) as f:
            for line in f.readlines():
                img_id, split_v = line.strip().split(" ")
                if split == "train" and int(split_v) == 1:
                    data_ids.append(int(img_id))
                elif split != "train" and int(split_v) == 0:
                    data_ids.append(int(img_id))
        
        data_paths = {}
        with open(os.path.join(root, "images.txt")) as f:
            for line in f.readlines():
                img_id, end_path = line.strip().split(" ")
                data_paths[int(img_id)] = end_path
        
        lbl_map = {}
        with open(os.path.join(root, "image_class_labels.txt")) as f:
            for line in f.readlines():
                img_id, lbl = line.strip().split(" ")
                lbl_map[int(img_id)] = int(lbl) - 1

        self.paths = []
        self.lbls = []
        for img_id in data_ids:
            self.paths.append(os.path.join(root, "images", data_paths[img_id]))
            self.lbls.append(lbl_map[img_id])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        lbl = self.lbls[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, lbl