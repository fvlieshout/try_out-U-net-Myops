import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BB_model(nn.Module):
    def __init__(self) -> None:
        super(BB_model, self).__init__()
        self.conv3D1 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.conv3D2 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.conv3D3 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers)
        self.bb_layer = nn.Linear(512, 4)
        self.relu = nn.ReLU()
    
    def forward(self, stacked_input):
        stacked_input = self.conv3D1(stacked_input.unsqueeze(1))
        stacked_input = self.conv3D2(stacked_input)
        stacked_input = self.conv3D3(stacked_input)[0]
        h = stacked_input.shape[1]
        stacked_x = torch.zeros(h, 512)
        for i in range(h):
            x = stacked_input[:, i, :, :]
            x = torch.unsqueeze(x, 0)
            x = torch.repeat_interleave(x, 3, 1)
            x = self.features1(x)
            x = self.relu(x)
            x = nn.AdaptiveAvgPool2d((1,1))(x)
            stacked_x[i] = torch.squeeze(x)
        x = torch.unsqueeze(torch.mean(stacked_x, dim=0), 0).to(device)
        x = self.bb_layer(x)
        return x
        