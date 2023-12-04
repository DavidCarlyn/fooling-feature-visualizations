
"""
Reimplementation of real vs. visualization classifier as specified in Appendix C.1 in
the paper: DON’T TRUST YOUR EYES: ON THE (UN)RELIABILITY OF FEATURE VISUALIZATIONS
"""
import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import ImageFolder


from models.utils import load_pretrained_model

def train_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def get_training_data(imagenet_data, batch_size):
    train_dset = ImageFolder(imagenet_data, transform=train_transforms())

    train_dloader = DataLoader(train_dset, num_workers=8, batch_size=batch_size, shuffle=True)
    return train_dloader

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--imagenet_data", type=str, default="/local/scratch/datasets/imagenet/images/train/")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=64)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    train_dloader, _ = get_training_data(args.imagenet_data, args.batch_size)
    model = load_pretrained_model(args.model, num_classes=1000).cuda()
    model.eval()

    # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
    silent_activations = set()
    def check_silent():
        def hook(module, input, output):
            print(module, output)
        return hook
    
    hooks = {}
    for name, module in tqdm(model.named_modules(), desc="Adding hooks to all modules"):
        hooks[name] = module.register_forward_hook(check_silent)

    for batch in tqdm(train_dloader, desc="Checking for silent units"):
        imgs, _ = batch
        imgs = imgs.cuda()
        _ = model(imgs)