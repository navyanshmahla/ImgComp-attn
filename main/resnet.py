import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from attention.channel import *
from attention.spatial import *


# Define the residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.ca = ChannelGate(out_channels)
        self.sa = SpatialGate()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.ca(out)
        out = self.sa(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out

# Define the ResNet architecture
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_channels=64, num_classes=10):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.residual_blocks = self._make_residual_blocks(num_blocks, num_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_classes)

    def _make_residual_blocks(self, num_blocks, num_channels):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(num_channels, num_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.residual_blocks(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = ResNet(num_blocks=30)

print(model)

