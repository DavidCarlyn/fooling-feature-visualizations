import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.utils import load_pretrained_model
from datasets.cub import CUB_Dataset, train_transforms, test_transforms

import lightning.pytorch as pl
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only

class CUBLightningModel(pl.LightningModule):
    def __init__(self, args, arch):
        super().__init__()
        self.args = args
        self.arch = arch
        self.loss_fn = nn.CrossEntropyLoss()

    def calc_correct(self, out, lbls):
        _, preds = torch.max(out, dim=1)

        correct = (preds == lbls).sum().item()
        return correct

    def validation_step(self, batch, batch_idx):
        imgs, lbls = batch
        out = self.arch(imgs)
        loss = self.loss_fn(out, lbls)
        correct = self.calc_correct(out, lbls)

        self.log_dict({
            "val_acc" : correct / len(imgs),
            "val_loss" : loss.item(),
        }, on_epoch=True, on_step=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        imgs, lbls = batch
        out = self.arch(imgs)
        loss = self.loss_fn(out, lbls)
        correct = self.calc_correct(out, lbls)

        self.log_dict({
            "train_acc" : correct / len(imgs),
            "train_loss" : loss.item(),
        }, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.arch.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-5)
        return optimizer


def load_data(data_root, batch_size):
    tr_trans = train_transforms()
    te_trans = test_transforms()
    
    train_dset = CUB_Dataset(data_root, split="train", transforms=tr_trans)
    test_dset = CUB_Dataset(data_root, split="test", transforms=te_trans)

    train_dloader = DataLoader(train_dset, num_workers=4, batch_size=batch_size, shuffle=True)
    test_dloader = DataLoader(test_dset, num_workers=4, batch_size=batch_size, shuffle=False)

    return train_dloader, test_dloader

def train_model(model, train_dloader, val_dloader, epochs=100, lr=0.001):
    optimizer = optim.SGD(lr=lr, params=model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Training", colour="yellow", position=0):
        model.train()
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
    parser.add_argument("--data_root", type=str, default="/local/scratch/cv_datasets/CUB_200_2011")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--outdir", type=str, default="data/models")
    
    return parser.parse_args()

class MetricTracker(pl.Callback):

    def __init__(self, args):
        self.args = args

    @rank_zero_only
    def on_train_epoch_end(self, trainer, module):
        out_str = ""
        for key in trainer.logged_metrics:
            out_str += f"{key}: {trainer.logged_metrics[key].item()} | "
        print(out_str)

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    model = load_pretrained_model(args.model, num_classes=200).cuda()
    train_dloader, val_dloader = load_data(args.data_root, args.batch_size)

    lightning_model = CUBLightningModel(args, model)

    mt = MetricTracker(args=args)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=args.epochs,
        log_every_n_steps=1,
        callbacks=[mt])

    trainer.fit(lightning_model, train_dloader, val_dloader)


    #model = train_model(model, train_dloader, val_dloader, epochs=args.epochs, lr=args.lr)

    save_model(lightning_model.arch, os.path.join(args.outdir, f"cub_{args.model}.pt"))