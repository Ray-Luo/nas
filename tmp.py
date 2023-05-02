import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_LATENCY = 10

class InvertedResidualBase(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expansion=6,
        out_channel_mask=None,
    ):
        super(InvertedResidualBase, self).__init__()
        assert stride in [1, 2]

        hidden_dim = expansion * inp

        self.identity = stride == 1 and inp == oup

        self.out_channel_mask = out_channel_mask
        self.expansion_mask = nn.Parameter(torch.ones(hidden_dim, requires_grad=True), requires_grad=True)

        self.act_mask = nn.Parameter(torch.zeros(1, requires_grad=True))

        self.relu = nn.ReLU(inplace=False)

        self.se_mask = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.id = nn.Identity()

        self.pw = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        latency = 0

        expansion_mask = torch.sigmoid(self.expansion_mask).view(
            1, -1, 1, 1
        )
        expansion_sum = torch.sum(expansion_mask)

        # point_wise conv

        out = self.pw(x)
        latency += x.shape[1] * expansion_sum * BASE_LATENCY # pw
        latency += expansion_sum * BASE_LATENCY # bn
        latency += BASE_LATENCY # act

        out = self.pw_linear(out)

        return out, latency


import torch
import torch.nn as nn
import torch.nn.functional as F

def print_grad(grad):
    print("Gradient:", grad)

class ChannelMaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ChannelMaskedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask = nn.Parameter(torch.ones(in_channels), requires_grad=True)
        # self.mask.register_hook(print_grad)

    def forward(self, x):
        # Normalize the mask to have values between 0 and 1
        mask = torch.sigmoid(self.mask)

        # Reshape and expand the mask to match the input tensor shape
        mask_reshaped = mask.view(
            1, -1, 1, 1
        )

        latency = torch.sum(mask_reshaped)

        # Apply the mask to the input tensor
        x_masked = x * mask_reshaped

        # Pass the masked input through the convolution layer
        return self.conv(x_masked), latency

class ThreeLayerConvNet(nn.Module):
    def __init__(self):
        super(ThreeLayerConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = ChannelMaskedConv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x, latency = self.conv2(x)
        x = F.relu(x)
        x = F.relu(self.conv3(x))
        return x, latency


class RepeatMask(nn.Module):
    def __init__(self, num_classes):
        super(RepeatMask, self).__init__()
        one_hot = torch.zeros(num_classes)
        one_hot[num_classes-1] = 1
        print(one_hot)
        self.p = nn.Parameter(torch.tensor(one_hot), requires_grad=True)
        self.temperature = 1.0

    def forward(self, hard=False):
        logits = F.gumbel_softmax(self.p, tau=self.temperature, hard=hard)
        print(logits)
        one_hot_index = torch.argmax(logits)
        # return one_hot_index
        return one_hot_index

if 0:
# Instantiate the network
    # model = ThreeLayerConvNet().cuda()
    model = InvertedResidualBase(3,3,1).cuda()


    target = torch.randn(1,3,512,512).cuda()
    input = torch.randn(1, 3, 512, 512).cuda()
    # model = InvertedResidualBase(3,3,1).cuda()
    # out, latency_original = model(input)

    import torch.optim as optim

    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    criterion = nn.L1Loss()
    num_epochs = 1000
    weight = 1e-3

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        out, latency = model(input)

        loss = criterion(out, target)
        loss += latency * weight


        # loss = latency * weight

        # print("Epoch: {}, Loss: {}, Lat: {}, Ori_lat: {}".format(epoch, loss.item(), latency.item(), latency_original.item()))
        loss.backward()
        # print("******", model.conv2.mask.grad)
        print("******", latency.item(), loss.item())
        # print("******", model.expansion_mask.grad)
        # print("******", loss.item())
        optimizer.step()

# import torch
# a = torch.randn((6,), requires_grad=True)
# x = torch.randn((2, 3), requires_grad=True)
# y = x.view((6,)).expand_as(a)
# z = torch.sum(y)
# z.backward()

# print(x.grad)

# model = RepeatMask(3)
# print(model.forward())

import torch.nn.functional as F

x = torch.tensor([0, 2, 1])
one_hot = F.one_hot(x, num_classes=3)

print(one_hot)
