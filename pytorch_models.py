import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()

        model_ft = models.googlenet(pretrained=True)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, 512)

        self.backbone = model_ft

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        model_ft = models.resnet18(pretrained=True)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, 512)

        self.backbone = model_ft

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        return x


class ShuffleNet(nn.Module):
    def __init__(self):
        super(ShuffleNet, self).__init__()

        model_ft = models.shufflenetv2.shufflenet_v2_x2_0(pretrained=True)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, 512)
        self.backbone = model_ft

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        model_ft = models.mobilenetv2.mobilenet_v2(pretrained=True)
        model_ft.classifier[1] = nn.Linear(model_ft.classifier[1].in_features, 512)

        self.backbone = model_ft

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        downsampler = lambda n: nn.Sequential(
            nn.Conv2d(n, 2 * n, kernel_size=1, stride=(2, 2), bias=False),
            nn.BatchNorm2d(2 * n),
        )

        basic_block = lambda n: models.resnet.BasicBlock(
            n, 2 * n, 2, downsample=downsampler(n)
        )

        # 1. Change model architecture
        model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # model_ft.layer3 = basic_block(128)
        model_ft.layer4 = basic_block(256)

        # 2. Change model's last layer for 512 dimensional embedding

        model_ft.fc = nn.Linear(model_ft.fc.in_features, 512)

        self.backbone = model_ft

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        return x
