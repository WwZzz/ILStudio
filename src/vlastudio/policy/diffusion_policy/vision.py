"""DP's torchvision backbone and spatial keypoint pooling.

Matches the ResNet18Conv/SpatialSoftmax parameter layout used by existing DP
checkpoints, without depending on RoboMimic's simulator and EGL build tooling.
Architecture reference: robomimic v0.3.0, models/base_nets.py.
"""
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Conv(nn.Module):
    def __init__(self, input_channel=3, pretrained=False, input_coord_conv=False):
        super().__init__()
        if input_coord_conv:
            raise ValueError("DP's image encoder does not use coordinate convolutions")
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        if input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, 7, stride=2, padding=3, bias=False)
        # Preserve the original checkpoint names, including the 'nets' prefix.
        self.nets = nn.Sequential(*list(net.children())[:-2])

    def forward(self, image):
        return self.nets(image)


class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp=32, temperature=1.0,
                 learnable_temperature=False, noise_std=0.0):
        super().__init__()
        channels, self.height, self.width = input_shape
        self.num_kp = channels if num_kp is None else num_kp
        self.nets = None if num_kp is None else nn.Conv2d(channels, num_kp, 1)
        self.noise_std = noise_std
        value = torch.full((1,), float(temperature))
        if learnable_temperature:
            self.temperature = nn.Parameter(value)
        else:
            self.register_buffer("temperature", value)
        x, y = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        self.register_buffer("pos_x", torch.from_numpy(x.reshape(1, -1)).float())
        self.register_buffer("pos_y", torch.from_numpy(y.reshape(1, -1)).float())

    def forward(self, feature):
        if feature.shape[-2:] != (self.height, self.width):
            raise ValueError("SpatialSoftmax feature size differs from its configured input")
        if self.nets is not None:
            feature = self.nets(feature)
        attention = (feature.reshape(-1, self.height * self.width) / self.temperature).softmax(-1)
        x = (attention * self.pos_x).sum(-1, keepdim=True)
        y = (attention * self.pos_y).sum(-1, keepdim=True)
        keypoints = torch.cat((x, y), dim=-1).reshape(-1, self.num_kp, 2)
        if self.training:
            keypoints = keypoints + torch.randn_like(keypoints) * self.noise_std
        return keypoints
