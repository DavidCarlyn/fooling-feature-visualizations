import torch

from lucent.optvis import render
from lucent.modelzoo import inceptionv1

device = torch.device("cuda:0")
model = inceptionv1(pretrained=True)
model.to(device).eval()

render.render_vis(model, "softmax2_pre_activation_matmul:1005")