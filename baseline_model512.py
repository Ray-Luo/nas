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
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
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
            nn.ReLU(inplace=True),
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
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_c, out_c, kernel_size=1),
    )


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride, expansion, use_se=False, use_hs=False):
        super(InvertedResidualBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = expansion * inp

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
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
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
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
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class UpInvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride=2, expansion=6, use_se=False, use_hs=False):
        super(UpInvertedResidualBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = expansion * inp

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
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
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
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
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class UNetMobileNetv3(nn.Module):
    """
    Modified UNet with inverted residual block and depthwise seperable convolution
    """

    def __init__(self, out_size):
        super(UNetMobileNetv3, self).__init__()
        self.out_size = out_size

        expansion = 1

        # encoding arm
        self.irb_bottleneck1 = self.irb_bottleneck(3, 16, 1, 2, 1)
        self.irb_bottleneck2 = self.irb_bottleneck(16, 32, 1, 2, expansion)
        self.irb_bottleneck3 = self.irb_bottleneck(32, 48, 1, 2, expansion)
        self.irb_bottleneck4 = self.irb_bottleneck(48, 96, 1, 2, expansion)
        self.irb_bottleneck5 = self.irb_bottleneck(96, 128, 1, 2, expansion)
        self.irb_bottleneck6 = self.irb_bottleneck(128, 256, 1, 2, expansion)
        self.irb_bottleneck7 = self.irb_bottleneck(256, 320, 1, 1, expansion)
        # decoding arm
        self.D_irb1 = self.irb_bottleneck(320, 128, 1, 2, expansion, True)
        self.D_irb2 = self.irb_bottleneck(128, 96, 1, 2, expansion, True)
        self.D_irb3 = self.irb_bottleneck(96, 48, 1, 2, expansion, True)
        self.D_irb4 = self.irb_bottleneck(48, 32, 1, 2, expansion, True)
        self.D_irb5 = self.irb_bottleneck(32, 16, 1, 2, expansion, True)
        self.D_irb6 = self.irb_bottleneck(16, 3, 1, 2, expansion, True)

    def depthwise_conv(self, in_c, out_c, k=3, s=1, p=0):
        """
        optimized convolution by combining depthwise convolution and
        pointwise convolution.
        """
        conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=k, padding=p, groups=in_c, stride=s),
            nn.BatchNorm2d(num_features=in_c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=1),
        )
        return conv

    def irb_bottleneck(self, in_c, out_c, n, s, t, d=False):
        """
        create a series of inverted residual blocks.
        """
        convs = []
        if d:
            xx = UpInvertedResidualBlock(in_c, out_c, s, t)
            convs.append(xx)
            if n > 1:
                for _ in range(1, n):
                    xx = UpInvertedResidualBlock(out_c, out_c, 1, t)
                    convs.append(xx)
            conv = nn.Sequential(*convs)
        else:
            xx = InvertedResidualBlock(in_c, out_c, s, t)
            convs.append(xx)
            if n > 1:
                for _ in range(1, n):
                    xx = InvertedResidualBlock(out_c, out_c, 1, t)
                    convs.append(xx)
            conv = nn.Sequential(*convs)
        return conv

    def forward(self, x):
        x1 = x
        x2 = self.irb_bottleneck1(x1)
        x3 = self.irb_bottleneck2(x2)
        x4 = self.irb_bottleneck3(x3)
        x5 = self.irb_bottleneck4(x4)
        x6 = self.irb_bottleneck5(x5)
        x7 = self.irb_bottleneck6(x6)
        x8 = self.irb_bottleneck7(x7)

        # Right arm / Decoding arm with skip connections
        d1 = self.D_irb1(x8) + x6
        d2 = self.D_irb2(d1) + x5
        d3 = self.D_irb3(d2) + x4
        d4 = self.D_irb4(d3) + x3
        d5 = self.D_irb5(d4) + x2
        d6 = self.D_irb6(d5)
        return d6


import torch
target = torch.randn(1,3,512,512)#.cuda()
input = torch.randn(1, 3, 512, 512)#.cuda()
model = UNetMobileNetv3(512)#.cuda()

output = model(input)
print(output.size())

num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters: ", num_params)

from thop import profile
macs, params = profile(model, inputs=(input, ))
print(macs)
