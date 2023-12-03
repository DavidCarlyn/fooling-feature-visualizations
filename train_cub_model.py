import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.utils import load_pretrained_model
from datasets.cub import CUB_Dataset, train_transforms, test_transforms


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

    model = load_pretrained_model(args.model, num_classes=200).cuda()
    train_dloader, val_dloader = load_data(args.data_root, args.batch_size)
    model = train_model(model, train_dloader, val_dloader, epochs=args.epochs, lr=args.lr)
    save_model(model, os.path.join(args.outdir, f"cub_{args.model}.pt"))