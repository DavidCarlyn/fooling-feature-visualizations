
"""
Reimplementation of real vs. visualization classifier as specified in Appendix C.1 in
the paper: DON’T TRUST YOUR EYES: ON THE (UN)RELIABILITY OF FEATURE VISUALIZATIONS
"""
import os
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from lucent.modelzoo.inceptionv1 import helper_layers
from adv_lib.attacks import ddn

from models.utils import load_pretrained_model
from datasets.cub import CUB_Dataset, train_transforms

def imgnet_train_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def get_training_data(train_data, batch_size, train_dset="imgnet"):
    if train_dset == "imgnet":
        train_dset = ImageFolder(train_data, transform=imgnet_train_transforms())
    elif train_dset == "cub":
        train_dset = CUB_Dataset(train_data, split="train", transforms=train_transforms())

    train_dloader = DataLoader(train_dset, num_workers=8, batch_size=batch_size, shuffle=True)
    return train_dloader

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, default="/local/scratch/datasets/imagenet/images/train/")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=str, default="imgnet")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--outdir", type=str, default="data/results")
    parser.add_argument("--modeldir", type=str, default="data/models")
    parser.add_argument("--run_adversarial", action="store_true", default=False)
    parser.add_argument("--attack_sample_idxs", nargs='+', type=int, default=[7, 950, 1245, 3206, 4055])
    parser.add_argument("--attack_method", type=str, default="ddn")

    return parser.parse_args()

def save_images(org, adv, path):
    org_out = np.array(T.functional.to_pil_image(org[0]))
    for i in range(1, org.shape[0]):
        org_out = np.concatenate((org_out, np.array(T.functional.to_pil_image(org[i]))), axis=1)
    
    adv_out = np.array(T.functional.to_pil_image(adv[0]))
    for i in range(1, adv.shape[0]):
        adv_out = np.concatenate((adv_out, np.array(T.functional.to_pil_image(adv[i]))), axis=1)
    
    out = np.concatenate((org_out, adv_out), axis=0)
    Image.fromarray(out).save(path)

if __name__ == "__main__":
    args = get_args()

    train_dloader = get_training_data(args.train_data, args.batch_size, train_dset=args.pretrained)

    if args.pretrained == "imgnet":
        model = load_pretrained_model(args.model, num_classes=1000).cuda()
    elif args.pretrained == "cub":
        model = load_pretrained_model(args.model, num_classes=200).cuda()
        model.load_state_dict(torch.load(os.path.join(args.modeldir, f"cub_{args.model}.pt")))
    model.eval()

    # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
    silent_activations = {}
    module_counts = {}
    def check_silent(name):
        def hook(module, input, output):
            flat = output.view(len(output), -1)
            key_name = f"{name}_{module_counts[name]}"
            if key_name not in silent_activations:
                silent_activations[key_name] = torch.zeros(flat.shape[1], dtype=torch.bool).cuda()
            has_act = (flat != 0).sum(0) > 0
            silent_activations[key_name] = torch.logical_or(silent_activations[key_name], has_act)
            module_counts[name] += 1
            #print(name, output.view(len(output), -1).shape)
        return hook
    
    hooks = {}
    for name, module in tqdm(model.named_modules(), desc="Adding hooks to all modules"):
        def is_class_of(cls_defs):
            for cls_def in cls_defs:
                if isinstance(module, cls_def): return True
            return False
        
        hook_modules = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
                        nn.Linear, helper_layers.RedirectedReluLayer,
                        helper_layers.ReluLayer]
        if is_class_of(hook_modules):
            hooks[name] = module.register_forward_hook(check_silent(name))
            module_counts[name] = 0

    def compute_num_silent_units():
        silent_units = 0
        for name in silent_activations:
            silent_units += (~silent_activations[name]).sum()
        return silent_units

    with torch.no_grad():
        for batch in (pbar := tqdm(train_dloader)):
            imgs, _ = batch
            imgs = imgs.cuda()
            _ = model(imgs)

            silent_units = compute_num_silent_units()
            pbar.set_description(f"Slient Units: {silent_units}")

            for name in module_counts:
                module_counts[name] = 0

    out_str = f"Total # of silent units in {args.model}: {silent_units}"
    print(out_str)
    with open(os.path.join(args.outdir, f"silent_units_{args.pretrained}.txt"), 'a') as f:
        f.write(out_str + '\n')

    if args.run_adversarial and args.pretrained == "cub":
        def adv_transform():
            return T.Compose([
                T.CenterCrop((256, 256)),
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

        # Hack to reset module counts when creating adversarial
        class AdvCallback():
            def __init__(self):
                self.count = 0 
            def accumulate_line(self, a, b, c):
                self.count += 1
                if self.count == 3:
                    self.count = 0
                    for name in module_counts:
                        module_counts[name] = 0

            def update_lines(self):
                pass

        train_dset = CUB_Dataset(args.train_data, split="val", transforms=adv_transform())
        exs = None
        lbls = []
        for idx in args.attack_sample_idxs:
            ex, lbl = train_dset[idx]
            ex = ex.unsqueeze(0).cuda()
            if exs is None:
                exs = ex
            else:
                exs = torch.cat((exs, ex), dim=0)
            lbls.append(lbl)
        lbls = torch.tensor(lbls).cuda()
        if args.attack_method == "ddn":
            callback = AdvCallback()
            adv_sample = ddn(model=model, inputs=exs, labels=lbls, steps=300, callback=callback)

        silent_units = compute_num_silent_units()
        out_str = f"Total # of silent units in {args.model} after {args.attack_method} attack: {silent_units}"
        print(out_str)
        with open(os.path.join(args.outdir, f"silent_units_{args.pretrained}_adv.txt"), 'a') as f:
            f.write(out_str + '\n')

        with torch.no_grad():
            org_out = model(exs)
            _, org_preds = torch.max(org_out, dim=1)
            org_correct = (org_preds == lbls).int()
            adv_out = model(adv_sample)
            _, adv_preds = torch.max(adv_out, dim=1)
            adv_correct = (adv_preds == lbls).int()
            print(org_correct)
            print(adv_correct)

        save_images(exs, adv_sample, os.path.join(args.outdir, "adv_ex.png"))

