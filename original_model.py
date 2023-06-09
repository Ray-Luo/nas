import torch.nn as nn

BASE_LATENCY = 10

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

        self.latency = self.get_latency(inp, oup, hidden_dim)

    def get_latency(self, inp, oup, hidden_dim):
        latency = 0
        if inp != hidden_dim:
            latency += inp * hidden_dim * BASE_LATENCY # pw
            latency += hidden_dim * BASE_LATENCY # bn
            latency += BASE_LATENCY # act
        latency += hidden_dim * hidden_dim * BASE_LATENCY # dw
        latency += hidden_dim * BASE_LATENCY # bn
        latency += BASE_LATENCY # act
        latency += BASE_LATENCY # se
        latency += hidden_dim * oup * BASE_LATENCY # pw-linear
        latency += oup * BASE_LATENCY # bn

        return latency


    def forward(self, x):
        if self.identity:
            return x + self.conv(x), self.latency
        else:
            return self.conv(x), self.latency


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

        self.latency = self.get_latency(inp, oup, hidden_dim)

    def get_latency(self, inp, oup, hidden_dim):
        latency = 0
        if inp != hidden_dim:
            latency += inp * hidden_dim * BASE_LATENCY # pw
            latency += hidden_dim * BASE_LATENCY # bn
            latency += BASE_LATENCY # act
        latency += hidden_dim * hidden_dim * BASE_LATENCY # dw
        latency += hidden_dim * BASE_LATENCY # bn
        latency += BASE_LATENCY # act
        latency += BASE_LATENCY # se
        latency += hidden_dim * oup * BASE_LATENCY # pw-linear
        latency += oup * BASE_LATENCY # bn

        return latency

    def forward(self, x):
        if self.identity:
            return x + self.conv(x), self.latency
        else:
            return self.conv(x), self.latency


class UNetMobileNetv3(nn.Module):
    """
    Modified UNet with inverted residual block and depthwise seperable convolution
    """

    def __init__(self, out_size):
        super(UNetMobileNetv3, self).__init__()
        self.out_size = out_size

        expansion = 6

        # encoding arm
        self.conv3x3 = self.depthwise_conv(3, 16, p=1, s=2)
        self.irb_bottleneck1 = self.irb_bottleneck(16, 24, 1, 1, 1)
        self.irb_bottleneck2 = self.irb_bottleneck(24, 32, 2, 2, expansion)
        self.irb_bottleneck3 = self.irb_bottleneck(32, 48, 3, 2, expansion)
        self.irb_bottleneck4 = self.irb_bottleneck(48, 96, 4, 2, expansion)
        self.irb_bottleneck5 = self.irb_bottleneck(96, 128, 4, 2, expansion)
        self.irb_bottleneck6 = self.irb_bottleneck(128, 256, 3, 1, expansion)
        self.irb_bottleneck7 = self.irb_bottleneck(256, 320, 1, 2, expansion)
        # decoding arm
        self.D_irb1 = self.irb_bottleneck(320, 128, 1, 2, expansion, True)
        self.D_irb2 = self.irb_bottleneck(128, 96, 1, 2, expansion, True)
        self.D_irb3 = self.irb_bottleneck(96, 48, 1, 2, expansion, True)
        self.D_irb4 = self.irb_bottleneck(48, 32, 1, 2, expansion, True)
        self.D_irb5 = self.irb_bottleneck(32, 24, 1, 2, expansion, True)
        self.D_irb6 = self.irb_bottleneck(24, 16, 1, 2, expansion, True)
        self.D_irb7 = self.irb_bottleneck(16, 3, 1, 1, expansion, False)
        # self.DConv4x4 = nn.ConvTranspose2d(32, 16, 4, 2, 1, groups=16, bias=False)
        # Final layer: output channel number can be changed as per the usecase
        # self.conv1x1_decode = nn.Conv2d(16, 3, kernel_size=1, stride=1)

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

    def irb_forward(self, blocks, x):
        latency = 0
        for i in range(len(blocks)):
            block = blocks[i]
            x, cur_lat = block(x)
            latency += cur_lat

        return x, latency

    def forward(self, x):
        x1 = self.conv3x3(x)  # (32, 256, 256)
        x2, lat2 = self.irb_forward(self.irb_bottleneck1, x1)  # (16,256,256) s1
        x3, lat3 = self.irb_forward(self.irb_bottleneck2, x2)  # (24,128,128) s2
        x4, lat4 = self.irb_forward(self.irb_bottleneck3, x3)  # (32,64,64) s3
        x5, lat5 = self.irb_forward(self.irb_bottleneck4, x4)  # (64,32,32)
        x6, lat6 = self.irb_forward(self.irb_bottleneck5, x5)  # (96,16,16) s4
        x7, lat7 = self.irb_forward(self.irb_bottleneck6, x6)  # (160,16,16)
        x8, lat8 = self.irb_forward(self.irb_bottleneck7, x7)  # (240,8,8)

        # Right arm / Decoding arm with skip connections
        d1, lat9 = self.irb_forward(self.D_irb1, x8)
        d1 += x6
        d2, lat10 = self.irb_forward(self.D_irb2, d1)
        d2 += x5
        d3, lat11 = self.irb_forward(self.D_irb3, d2)
        d3 += x4
        d4, lat12 = self.irb_forward(self.D_irb4, d3)
        d4 += x3
        d5, lat13 = self.irb_forward(self.D_irb5, d4)
        d5 += x2
        d6, lat14 = self.irb_forward(self.D_irb6, d5)
        d7, lat15 = self.irb_forward(self.D_irb7, d6)
        return d7, sum([lat2,lat3,lat4,lat5,lat6,lat7,lat8,lat9,lat10,lat11,lat12,lat13,lat14,lat15,])


import torch
input = torch.randn(1, 3, 512, 512)
model = UNetMobileNetv3(512)
out, latency = model(input)
print(out.shape, latency)

um_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of model parameters: ", um_params)

"""
13,295,908, 344,216,100
            150,450,000
 5324810
   12954
"""
