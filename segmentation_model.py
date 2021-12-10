import torch
from torch.nn.modules.container import ModuleList
from torch.nn.modules.conv import ConvTranspose2d
import torchvision.models as models
from torchvision.transforms import CenterCrop
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UnetBlock(nn.Module):
    def __init__(self, inChannels, outChannels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outChannels, outChannels, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class UnetEncoder(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64)) -> None:
        super().__init__()
        self.encBlocks = nn.ModuleList(
            [UnetBlock(channels[i], channels[i+1])
            for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        blockOutputs = []

        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        return blockOutputs

class UnetDecoder(nn.Module):
    def __init__(self, channels=(64, 32, 16)) -> None:
        super().__init__()
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i+1], 2, 2)
            for i in range(len(channels) - 1)]
        )
        self.decBlocks = nn.ModuleList(
            [UnetBlock(channels[i], channels[i+1])
            for i in range(len(channels) - 1)]
        )

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.decBlocks[i](x)
        return x

    def crop(self, encFeatures, x):
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H,W])(encFeatures)
        return encFeatures

class UNet(nn.Module):
    def __init__(self, encChannels=(1,16,32,64), decChannels = (64, 32, 16), nbClasses=1, retainDim=True, outSize=(512, 512)) -> None:
        super().__init__()
        self.encoder = UnetEncoder(encChannels)
        self.decoder = UnetDecoder(decChannels)

        self.head = nn.Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        encFeatures = self.encoder(x)
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
        map = self.head(decFeatures)
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        return map
