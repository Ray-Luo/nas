import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


class ELSR_base(nn.Module):
    def __init__(self):
        super(ELSR_base, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(6, 6, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(6, 6, 3, padding=1),
        )

class ESLR_x2(ELSR_base):
    def __init__(self):
        super().__init__()

        self.last_conv = nn.Conv2d(6, 12, 3, padding=1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv1(x)
        bottleneck = self.bottleneck(x)
        x = bottleneck + x
        x = self.last_conv(x)
        x = self.ps(x)

        return x

class ESLR_x4(ELSR_base):
    def __init__(self):
        super().__init__()

        self.last_conv = nn.Conv2d(6, 48, 3, padding=1)
        self.ps = nn.PixelShuffle(4)

    def forward(self, x):
        x = self.conv1(x)
        bottleneck = self.bottleneck(x)
        x = bottleneck + x
        x = self.last_conv(x)
        x = self.ps(x)

        return x


import torch
target = torch.randn(1,3,512,512)#.cuda()
input = torch.randn(1, 3, 512, 512)#.cuda()
model = ESLR_x4()#.cuda()

output = model(input)
print(output.size())

num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters: ", num_params)

from thop import profile
macs, params = profile(model, inputs=(input, ))
print("GFlops is {}".format(macs / 1e9))
