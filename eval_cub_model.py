import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.cub import CUB_Dataset, test_transforms
from models.utils import load_pretrained_model

def load_data(data_root, batch_size):
    test_dset = CUB_Dataset(data_root, split="test", transforms=test_transforms())

    test_dloader = DataLoader(test_dset, num_workers=4, batch_size=batch_size, shuffle=False)

    return test_dloader

def eval_model(model, val_dloader):
    loss_fn = nn.CrossEntropyLoss()
    total_val_correct = 0
    total_val = 0
    total_val_loss = 0
    with torch.no_grad():
        model.eval()
        for imgs, lbls in tqdm(val_dloader, desc="Validating", colour="blue"):
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            out = model(imgs)
            loss = loss_fn(out, lbls)

            total_val_loss += loss.item()
            _, preds = torch.max(out, dim=1)
            total_val_correct += (preds == lbls).sum().item()
            total_val += len(imgs)
        
        total_val_loss /= len(val_dloader)
        
    return total_val_loss, round(total_val_correct/total_val * 100, 2)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--data_root", type=str, default="E:\\datasets\\CUB_200_2011")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--outdir", type=str, default="data/models")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_path = os.path.join(args.outdir, f"cub_{args.model}.pt")
    weights = torch.load(model_path)

    model = load_pretrained_model(args.model, num_classes=200)
    model.load_state_dict(weights)
    model = model.cuda()

    val_dloader = load_data(args.data_root, args.batch_size)
    loss, acc = eval_model(model, val_dloader)

    print(f"{args.model} | Accuracy: {acc} | Loss: {loss}")