import os
from argparse import ArgumentParser
from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet
from torchvision.models import vgg
import torchvision.transforms as T

from lucent.modelzoo import inceptionv1

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

def load_model(model):
    num_classes = 200

    if model == "inceptionv1":
        net = inceptionv1(pretrained=True)
        net.softmax2_pre_activation_matmul = nn.Linear(1024, num_classes, bias=True)
    elif model == "resnet18":
        net = resnet.resnet18(resnet.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(512, num_classes)
    elif model == "resnet34":
        net = resnet.resnet34(resnet.ResNet34_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(512, num_classes)
    elif model == "resnet50":
        net = resnet.resnet50(resnet.ResNet50_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(2048, num_classes)
    elif model == "resnet101":
        net = resnet.resnet101(resnet.ResNet101_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(2048, num_classes)
    elif model == "resnet152":
        net = resnet.resnet152(resnet.ResNet152_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(2048, num_classes)
    elif model == "vgg11":
        net = vgg.vgg11(vgg.VGG11_Weights.IMAGENET1K_V1)
    elif model == "vgg13":
        net = vgg.vgg13(vgg.VGG13_Weights.IMAGENET1K_V1)
    elif model == "vgg16":
        net = vgg.vgg16(vgg.VGG16_Weights.IMAGENET1K_V1)
    elif model == "vgg19":
        net = vgg.vgg19(vgg.VGG19_Weights.IMAGENET1K_V1)
    else:
        NotImplementedError(f"{model} has not been implemented as an option. Please add an option and case.")

    if "vgg" in model:
        net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    return net


def load_data(data_root, batch_size):
    train_dset = CUB_Dataset(data_root, split="train", transforms=train_transforms())
    test_dset = CUB_Dataset(data_root, split="test", transforms=test_transforms())

    train_dloader = DataLoader(train_dset, num_workers=4, batch_size=batch_size, shuffle=True)
    test_dloader = DataLoader(test_dset, num_workers=4, batch_size=batch_size, shuffle=False)

    return train_dloader, test_dloader

def train_model(model, train_dloader, val_dloader, epochs=100, lr=0.001):
    optimizer = optim.SGD(lr=lr, params=model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Training", colour="yellow", position=0):
        total_train_loss = 0
        total_train_correct = 0
        total_val_correct = 0
        total_train = 0
        total_val = 0
        total_val_loss = 0
        for imgs, lbls in tqdm(train_dloader, desc="Training Loop", colour="green", position=1, leave=False):
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            out = model(imgs)
            loss = loss_fn(out, lbls)

            total_train_loss += loss.item()
            _, preds = torch.max(out, dim=1)
            total_train_correct += (preds == lbls).sum().item()
            total_train += len(imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_train_loss /= len(train_dloader)

        with torch.no_grad():
            model.eval()
            for imgs, lbls in tqdm(val_dloader, desc="Validation Loop", colour="blue", position=1, leave=False):
                imgs = imgs.cuda()
                lbls = lbls.cuda()
                out = model(imgs.cuda())
                loss = loss_fn(out, lbls.cuda())

                total_val_loss += loss.item()
                _, preds = torch.max(out, dim=1)
                total_val_correct += (preds == lbls).sum().item()
                total_val += len(imgs)

            total_val_loss /= len(val_dloader)

        print(f"Epoch {epoch} | training loss: {total_train_loss} | validation loss: {total_val_loss}")
        print(f"Epoch {epoch} | training acc: {round(total_train_correct/total_train * 100, 2)}% | validation acc: {round(total_val_correct/total_val * 100, 2)}%")

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--data_root", type=str, default="E:\\datasets\\CUB_200_2011")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--outdir", type=str, default="data/models")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    model = load_model(args.model).cuda()
    train_dloader, val_dloader = load_data(args.data_root, args.batch_size)
    model = train_model(model, train_dloader, val_dloader, epochs=args.epochs, lr=args.lr)
    save_model(model, os.path.join(args.outdir, f"cub_{args.model}.pt"))