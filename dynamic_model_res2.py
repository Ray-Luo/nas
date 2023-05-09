# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# Author: Lei Luo (luoleyouluole@fb.com)
import torch.nn as nn
import torch


import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    # ensure that all layers have a channel number that is divisible by 8

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=False),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), h_swish()
    )


def depthwise_conv(in_c, out_c, k=3, s=1, p=0):
    return nn.Sequential(
        nn.Conv2d(in_c, in_c, kernel_size=k, padding=p, groups=in_c, stride=s),
        nn.BatchNorm2d(num_features=in_c),
        nn.ReLU6(inplace=False),
        nn.Conv2d(in_c, out_c, kernel_size=1),
    )


class InvertedResidualBase(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        hidden_dim,
        use_hs=True,
        use_se=True,
    ):
        super(InvertedResidualBase, self).__init__()
        assert stride in [1, 2]

        self.oup = oup

        self.identity = stride == 1 and inp == oup

        self.if_pw = hidden_dim == inp

        self.act = h_swish() if use_hs else nn.ReLU(inplace=False)

        self.se = SELayer(hidden_dim) if use_se else nn.Identity()

        self.pw = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        # point_wise conv
        if not self.if_pw:
            pw_out = self.pw(x)
            out = self.act(pw_out)
        else:
            out = x

        # depthwise conv
        dw_out = self.dw(out)
        out = self.act(dw_out)
        out = self.se(out)

        # pointwise linear projection
        pwl_out = self.pw_linear(out)

        if self.identity:
            return x + pwl_out
        else:
            return pwl_out


class InvertedResidualBlock(InvertedResidualBase):
    def __init__(
        self,
        inp,
        oup,
        stride,
        hidden_dim,
        use_hs=True,
        use_se=True,
    ):
        super().__init__(
            inp,
            oup,
            stride,
            hidden_dim,
            use_hs,
            use_se,
        )
        self.dw = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride,
                (3 - 1) // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
        )


class UpInvertedResidualBlock(InvertedResidualBase):
    def __init__(
        self,
        inp,
        oup,
        stride,
        hidden_dim,
        use_hs=True,
        use_se=True,
    ):
        super().__init__(
            inp,
            oup,
            stride,
            hidden_dim,
            use_hs,
            use_se,
        )
        self.dw = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                4,
                stride,
                (4 - 1) // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
        )


class UNetMobileNetv3(nn.Module):
    """
    Modified UNet with inverted residual block and depthwise seperable convolution
    """

    def __init__(self, out_size):
        super(UNetMobileNetv3, self).__init__()
        self.out_size = out_size

        # encoding arm
        self.conv3x3 = self.depthwise_conv(3, 16, p=1, s=2)
        self.irb_bottleneck1 = nn.Sequential(
            InvertedResidualBlock(16, 21, 2, hidden_dim=93),
        )
        self.irb_bottleneck2 = nn.Sequential(
            InvertedResidualBlock(21, 31, 2, hidden_dim=127),
            InvertedResidualBlock(31, 31, 1, hidden_dim=127),
        )
        self.irb_bottleneck3 = nn.Sequential(
            InvertedResidualBlock(31, 62, 2, hidden_dim=189),
            InvertedResidualBlock(62, 62, 1, hidden_dim=189),
        )
        self.irb_bottleneck4 = nn.Sequential(
            InvertedResidualBlock(62, 84, 2, hidden_dim=382),
            InvertedResidualBlock(84, 84, 1, hidden_dim=382),
        )
        self.irb_bottleneck5 = nn.Sequential(
            InvertedResidualBlock(84, 169, 2, hidden_dim=507, use_hs=False),
        )
        self.irb_bottleneck6 = nn.Sequential(
            InvertedResidualBlock(169, 209, 1, hidden_dim=1039, use_hs=False),
        )

        # decoding arm
        self.D_irb1 = nn.Sequential(
            UpInvertedResidualBlock(209, 84, 2, hidden_dim=1291),
        )
        self.D_irb2 = nn.Sequential(
            UpInvertedResidualBlock(84, 62, 2, hidden_dim=551),
        )
        self.D_irb3 = nn.Sequential(
            UpInvertedResidualBlock(62, 31, 2, hidden_dim=377),
        )
        self.D_irb4 = nn.Sequential(
            UpInvertedResidualBlock(31, 21, 2, hidden_dim=189),
        )
        self.D_irb5 = nn.Sequential(
            UpInvertedResidualBlock(21, 16, 2, hidden_dim=124),
        )
        self.D_irb6 = nn.Sequential(
            UpInvertedResidualBlock(16, 16, 2, hidden_dim=93),
        )
        self.D_irb7 = nn.Sequential(
            InvertedResidualBlock(16, 3, 1, hidden_dim=60),
        )

    def depthwise_conv(self, in_c, out_c, k=3, s=1, p=0):
        conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=k, padding=p, groups=in_c, stride=s),
            nn.BatchNorm2d(num_features=in_c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=1),
        )
        return conv

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.irb_bottleneck1(x1)
        x3 = self.irb_bottleneck2(x2)
        x4 = self.irb_bottleneck3(x3)
        x5 = self.irb_bottleneck4(x4)
        x6 = self.irb_bottleneck5(x5)
        x7 = self.irb_bottleneck6(x6)

        # Right arm / Decoding arm with skip connections
        d1 = self.D_irb1(x7) + x5
        d2 = self.D_irb2(d1) + x4
        d3 = self.D_irb3(d2) + x3
        d4 = self.D_irb4(d3) + x2
        d5 = self.D_irb5(d4) + x1
        d6 = self.D_irb6(d5)
        d7 = self.D_irb7(d6)
        return d7



target = torch.randn(1,3,512,512)
input = torch.randn(1, 3, 512, 512)
model = UNetMobileNetv3(512)
out = model(input)
print(out.size())

from thop import profile
macs, params = profile(model, inputs=(input, ))
print(macs)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: ", num_params)
