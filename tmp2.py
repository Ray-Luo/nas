
import torch.nn as nn
import torch


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

        hidden_dim = expansion # * inp

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

        hidden_dim = expansion # * inp

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

        expansion = 6

        self.conv3x3 = self.depthwise_conv(3, 16, p=1, s=2)

        self.db1 = InvertedResidualBlock(inp=16, oup=17, stride=1, expansion=11, use_se=False, use_hs=False)

        self.db2 = InvertedResidualBlock(inp=17, oup=23, stride=2, expansion=102, use_se=False, use_hs=False)

        self.db3 = nn.Sequential(
            InvertedResidualBlock(inp=23, oup=34, stride=2, expansion=135, use_se=False, use_hs=False),
            InvertedResidualBlock(inp=34, oup=34, stride=1, expansion=200, use_se=False, use_hs=False)
        )

        self.db4 = nn.Sequential(
            InvertedResidualBlock(inp=34, oup=67, stride=2, expansion=198, use_se=False, use_hs=False),
            InvertedResidualBlock(inp=67, oup=67, stride=1, expansion=370, use_se=False, use_hs=False)
        )

        self.db5 = nn.Sequential(
            InvertedResidualBlock(inp=67, oup=83, stride=2, expansion=354, use_se=False, use_hs=False),
            InvertedResidualBlock(inp=83, oup=83, stride=1, expansion=439, use_se=False, use_hs=False)
        )

        self.db6 = nn.Sequential(
            InvertedResidualBlock(inp=83, oup=145, stride=1, expansion=416, use_se=False, use_hs=False),
            InvertedResidualBlock(inp=145, oup=145, stride=1, expansion=778, use_se=False, use_hs=False)
        )

        self.db7 = InvertedResidualBlock(inp=145, oup=173, stride=2, expansion=775, use_se=False, use_hs=False)

        self.ub1 = UpInvertedResidualBlock(inp=173, oup=83, stride=2, expansion=969, use_se=False, use_hs=False)

        self.ub2 = UpInvertedResidualBlock(inp=83, oup=67, stride=2, expansion=426, use_se=False, use_hs=False)

        self.ub3 = UpInvertedResidualBlock(inp=67, oup=34, stride=2, expansion=335, use_se=False, use_hs=False)

        self.ub4 = UpInvertedResidualBlock(inp=34, oup=23, stride=2, expansion=197, use_se=False, use_hs=False)

        self.ub5 = UpInvertedResidualBlock(inp=23, oup=17, stride=2, expansion=134, use_se=False, use_hs=False)

        self.ub6 = UpInvertedResidualBlock(inp=17, oup=16, stride=2, expansion=101, use_se=False, use_hs=False)

        self.ub7 = InvertedResidualBlock(inp=16, oup=3, stride=1, expansion=101, use_se=False, use_hs=False)


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
        # x1 = self.conv3x3(x)
        # x2 = self.irb_bottleneck1(x1)
        # x3 = self.irb_bottleneck2(x2)
        # x4 = self.irb_bottleneck3(x3)
        # x5 = self.irb_bottleneck4(x4)
        # x6 = self.irb_bottleneck5(x5)
        # x7 = self.irb_bottleneck6(x6)
        # x8 = self.irb_bottleneck7(x7)

        # # Right arm / Decoding arm with skip connections
        # d1 = self.D_irb1(x8) + x6
        # d2 = self.D_irb2(d1) + x5
        # d3 = self.D_irb3(d2) + x4
        # d4 = self.D_irb4(d3) + x3
        # d5 = self.D_irb5(d4) + x2
        # d6 = self.D_irb6(d5)
        # d7 = self.D_irb7(d6)
        # return d7

        x1 = self.conv3x3(x)
        x2 = self.db1(x1)
        x3 = self.db2(x2)
        x4 = self.db3(x3)
        x5 = self.db4(x4)
        x6 = self.db5(x5)
        x7 = self.db6(x6)
        x8 = self.db7(x7)



        # Right arm / Decoding arm with skip connections
        d1 = self.ub1(x8) + x6
        d2 = self.ub2(d1) + x5
        d3 = self.ub3(d2) + x4
        d4 = self.ub4(d3) + x3
        d5 = self.ub5(d4) + x2
        d6 = self.ub6(d5)
        d7 = self.ub7(d6)

        # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape, x7.shape, x8.shape)

        # print(d1.shape, d2.shape, d3.shape, d4.shape, d5.shape, d6.shape, d7.shape)

        return d7


target = torch.randn(1,3,512,512)
input = torch.randn(1, 3, 512, 512)
model = UNetMobileNetv3(512)
out = model(input)
print(out.shape)

import os
def to_onnx(model, input, output_dir):

    print("convert synthesis network...")
    mapping_net_trace = torch.jit.trace(model, input)
    torch.jit.save(mapping_net_trace, os.path.join(output_dir, "face_res.pt"))
    m_script = torch.jit.load(os.path.join(output_dir, "face_res.pt"))
    ops = torch.jit.export_opnames(m_script)
    print(ops)
    m_script._save_for_lite_interpreter(
        os.path.join(output_dir, "latest.ptl")
    )

to_onnx(model, input,  "/data/sandcastle/boxes/fbsource/fbcode/compphoto/media_quality/face_restoration/gfpgan/")
