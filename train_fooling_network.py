"""
Reimplementation of real vs. visualization classifier as specified in Appendix C.1 in
the paper: DON’T TRUST YOUR EYES: ON THE (UN)RELIABILITY OF FEATURE VISUALIZATIONS
"""
import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only

from models.real_v_vis import SimpleCNN
from datasets.fooling import FoolingDataset, train_transforms, test_transforms

def get_training_data(vis_data, imagenet_data, batch_size, small_test=False):
    train_dset = FoolingDataset(vis_data, imagenet_data, split="train", transforms=train_transforms(), small_test=small_test)
    val_dset = FoolingDataset(vis_data, imagenet_data, split="test", transforms=test_transforms(), small_test=small_test)

    train_dloader = DataLoader(train_dset, num_workers=8, batch_size=batch_size, shuffle=True)
    val_dloader = DataLoader(val_dset, num_workers=8, batch_size=batch_size, shuffle=False)
    return train_dloader, val_dloader
            
def save_model(model, path):
    torch.save(model.state_dict(), path)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--vis_data", type=str, default="/local/scratch/david/visualization_images")
    parser.add_argument("--imagenet_data", type=str, default="/local/scratch/datasets/imagenet/images/train/")
    parser.add_argument("--outdir", type=str, default="data/models")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--small_test", action='store_true', default=False)

    return parser.parse_args()

class SimpleLightningModel(pl.LightningModule):
    def __init__(self, args, arch):
        super().__init__()
        self.args = args
        self.arch = arch
        self.loss_fn = nn.BCELoss()

    def calc_correct(self, out, lbls):
        out = out.clone().detach()
        out[out >= 0.5] = 1
        out[out < 0.5] = 0

        correct = (out[:, 0] == lbls).sum().item()
        return correct

    def validation_step(self, batch, batch_idx):
        imgs, lbls = batch
        out = self.arch(imgs)
        out = nn.functional.sigmoid(out)
        loss = self.loss_fn(out, lbls.unsqueeze(1).type(torch.float32))
        correct = self.calc_correct(out, lbls)

        self.log_dict({
            "val_acc" : correct / len(imgs),
            "val_loss" : loss.item(),
        }, on_epoch=True, on_step=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        imgs, lbls = batch
        out = self.arch(imgs)
        out = nn.functional.sigmoid(out)
        loss = self.loss_fn(out, lbls.unsqueeze(1).type(torch.float32))
        correct = self.calc_correct(out, lbls)

        self.log_dict({
            "train_acc" : correct / len(imgs),
            "train_loss" : loss.item(),
        }, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-5)
        return optimizer

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

    train_dloader, test_dloader = get_training_data(args.vis_data, args.imagenet_data, args.batch_size, args.small_test)

    classifier = SimpleCNN()
    lightning_model = SimpleLightningModel(args, classifier)

    mt = MetricTracker(args=args)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=8,
        log_every_n_steps=1,
        callbacks=[mt])

    trainer.fit(lightning_model, train_dloader, test_dloader)
    save_model(lightning_model.arch, os.path.join(args.outdir, "fooling_classifier.pt"))