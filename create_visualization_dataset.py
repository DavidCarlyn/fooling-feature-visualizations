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

import numpy as np
from PIL import Image

from lucent.optvis import render, param
from lucent.modelzoo import inceptionv1


from models.real_v_vis import SimpleCNN

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--vis_data", type=str, default="E:\\datasets\\visualization_images")

    return parser.parse_args()

def load_pretrained_inception_net():
    model = inceptionv1(pretrained=True)
    model = model.to("cuda:0").eval()
    return model

def create_visulization_images(incep_net, vis_data_root, recalculate=False):
    thresholds = np.logspace(2, 9, num=15, base=2, dtype=np.uint16)
    for i in tqdm(range(0, 1000), desc="Create visualization dataset", position=0):
        target = f"softmax2_pre_activation_matmul:{i}"
        #target = f"softmax2:{i}"
        cls_dir = os.path.join(vis_data_root, f"{i}")
        #if os.path.exists(cls_dir) and not recalculate: continue
        os.makedirs(cls_dir, exist_ok=True)
        param_f = lambda: param.image(128, batch=35)
        #for j in tqdm(range(35), desc=f"Creating {i} visualizations", position=1, leave=False):
        visuals = render.render_vis(incep_net, target, param_f=param_f, thresholds=thresholds, show_image=False, progress=False)
        for step, vis in enumerate(visuals):
            for batch, im in enumerate(vis):
                Image.fromarray((np.array(im)*255).astype(np.uint8)).save(os.path.join(cls_dir, f"sample_{batch}_logstep_{step}.png"))

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.vis_data, exist_ok=True)
    incep_net = load_pretrained_inception_net()
    create_visulization_images(incep_net, args.vis_data)
