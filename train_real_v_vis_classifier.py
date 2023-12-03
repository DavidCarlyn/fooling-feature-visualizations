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
    parser.add_argument("--imagenet_data", type=str, default="E:\\datasets\\imagenet")

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

def get_training_data(vis_data, imagenet_data):
    return None, None

"""
All training hyperparameters are specified as per paper
"""
def train_classifier(classifier, train_dloader, test_dloader):
    optimizer = SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(8):
        correct = 0
        total = 0
        total_loss = 0
        for imgs, lbls in tqdm(train_dloader, desc=f"Epoch {epoch}"):
            out = classifier(imgs)
            loss = loss_fn(out, lbls)

            _, preds = torch.max(out, dim=1)
            correct += (preds == lbls).sum()
            total += len(imgs)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backwards()
            optimizer.step()
        
        print(f"Epoch {epoch} | training accuracy: {round((correct / total) * 100, 2)}% | training loss: {round(total_loss / len(train_dloader))}")

    return classifier
            

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.vis_data, exist_ok=True)
    incep_net = load_pretrained_inception_net()
    create_visulization_images(incep_net, args.vis_data)
    #train_dloader, test_dloader, get_training_data(args.vis_data, args.imagenet_data)
    #classifier = SimpleCNN()
    #classifier = train_classifier(classifier, train_dloader, test_dloader)
