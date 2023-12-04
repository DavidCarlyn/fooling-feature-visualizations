import torch.nn as nn

from torchvision.models import resnet
from torchvision.models import vgg
from lucent.modelzoo import inceptionv1

def load_pretrained_model(model, num_classes=200):
    def replace_last_layer(net, module=None):
        if num_classes == 1000: return net
        if "inception" in model:
            net.softmax2_pre_activation_matmul = module
        elif "resnet" in model:
            net.fc = module
        elif "vgg" in model:
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

    if model == "inceptionv1":
        net = inceptionv1(pretrained=True)
        replace_last_layer(net, nn.Linear(1024, num_classes, bias=True))
    elif model == "resnet18":
        net = resnet.resnet18(resnet.ResNet18_Weights.IMAGENET1K_V1)
        net = replace_last_layer(nn.Linear(512, num_classes))
    elif model == "resnet34":
        net = resnet.resnet34(resnet.ResNet34_Weights.IMAGENET1K_V1)
        net = replace_last_layer(nn.Linear(512, num_classes))
    elif model == "resnet50":
        net = resnet.resnet50(resnet.ResNet50_Weights.IMAGENET1K_V1)
        net = replace_last_layer(nn.Linear(2048, num_classes))
    elif model == "resnet101":
        net = resnet.resnet101(resnet.ResNet101_Weights.IMAGENET1K_V1)
        net = replace_last_layer(nn.Linear(2048, num_classes))
    elif model == "resnet152":
        net = resnet.resnet152(resnet.ResNet152_Weights.IMAGENET1K_V1)
        net = replace_last_layer(nn.Linear(2048, num_classes))
    elif model == "vgg11":
        net = replace_last_layer(vgg.vgg11(vgg.VGG11_Weights.IMAGENET1K_V1))
    elif model == "vgg13":
        net = replace_last_layer(vgg.vgg13(vgg.VGG13_Weights.IMAGENET1K_V1))
    elif model == "vgg16":
        net = replace_last_layer(vgg.vgg16(vgg.VGG16_Weights.IMAGENET1K_V1))
    elif model == "vgg19":
        net = replace_last_layer(vgg.vgg19(vgg.VGG19_Weights.IMAGENET1K_V1))
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