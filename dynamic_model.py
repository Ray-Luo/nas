import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_LATENCY = 10

class RepeatMask(nn.Module):
    def __init__(self, num_classes):
        super(RepeatMask, self).__init__()
        self.num_classes = num_classes
        self.p = nn.Parameter(torch.randn(num_classes), requires_grad=True)
        self.temperature = 1.0

    def forward(self, hard=True):
        logits = F.gumbel_softmax(self.p, tau=self.temperature, hard=hard)
        one_hot_index = torch.argmax(logits)
        # return one_hot_index
        return self.num_classes


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


class InvertedResidualBase(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expansion=None,
        out_channel_mask=None,
    ):
        super(InvertedResidualBase, self).__init__()
        assert stride in [1, 2]

        hidden_dim = expansion * inp

        self.identity = stride == 1 and inp == oup

        self.out_channel_mask = out_channel_mask
        self.expansion_mask = nn.Parameter(torch.ones(hidden_dim, requires_grad=True))

        self.act_mask = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.hs = h_swish()
        self.relu = nn.ReLU(inplace=True)

        self.se_mask = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.se = SELayer(hidden_dim)
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
        act_mask = torch.round(torch.sigmoid(self.act_mask))
        se_mask = torch.round(torch.sigmoid(self.se_mask))
        out_channel_mask = torch.round(torch.sigmoid(self.out_channel_mask)).view(
            1, -1, 1, 1
        )
        expansion_mask = torch.round(torch.sigmoid(self.expansion_mask)).view(
            1, -1, 1, 1
        )
        if_pw = torch.sum(expansion_mask) == x.shape[1]

        # point_wise conv
        if not if_pw:
            out = self.pw(x)
            out = act_mask * self.hs(out) + (1 - act_mask) * self.relu(out)
            latency += x.shape[1] * torch.sum(expansion_mask) * BASE_LATENCY # pw
            latency += torch.sum(expansion_mask) * BASE_LATENCY # bn
            latency += BASE_LATENCY # act
        else:
            out = x

        # depthwise conv
        out = self.dw(out)
        out = act_mask * self.hs(out) + (1 - act_mask) * self.relu(out)
        latency += torch.sum(expansion_mask) * torch.sum(expansion_mask) * BASE_LATENCY # dw
        latency += torch.sum(expansion_mask) * BASE_LATENCY # bn
        latency += BASE_LATENCY # act
        out = self.se(out) * se_mask + (1 - se_mask) * self.id(out)
        latency += torch.sum(expansion_mask).item() * BASE_LATENCY * se_mask.item()  + (1 - se_mask).item() * BASE_LATENCY # se
        out = out * expansion_mask
        # pointwise linear projection
        out = self.pw_linear(out)
        latency += torch.sum(expansion_mask) * torch.sum(out_channel_mask) * BASE_LATENCY # pw-linear
        latency += torch.sum(out_channel_mask) * BASE_LATENCY # bn

        if self.identity:
            out = x + out

        out = out * out_channel_mask
        return out, latency


class InvertedResidualBlock(InvertedResidualBase):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expansion=None,
        out_channel_mask=None,
    ):
        super().__init__(
            inp,
            oup,
            stride,
            expansion,
            out_channel_mask,)
        hidden_dim = expansion * inp
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
        expansion,
        out_channel_mask=None,
    ):
        super().__init__(
            inp,
            oup,
            stride,
            expansion,
            out_channel_mask,)
        hidden_dim = expansion * inp
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

        expansion = 6

        # encoding arm
        self.conv3x3 = self.depthwise_conv(3, 16, p=1, s=2)

        self.out_channel_mask_1 = nn.Parameter(torch.ones(24, requires_grad=True))
        self.irb_bottleneck1 = self.irb_bottleneck(
            16, 24, 1, 1, expansion=1, out_channel_mask=self.out_channel_mask_1
        )

        self.repeat_mask_2 = RepeatMask(2)
        self.out_channel_mask_2 = nn.Parameter(torch.ones(32, requires_grad=True))
        self.irb_bottleneck2 = self.irb_bottleneck(
            24, 32, 2, 2, expansion, self.out_channel_mask_2
        )

        self.repeat_mask_3 = RepeatMask(3)
        self.out_channel_mask_3 = nn.Parameter(torch.ones(48, requires_grad=True))
        self.irb_bottleneck3 = self.irb_bottleneck(
            32, 48, 3, 2, expansion, self.out_channel_mask_3
        )

        self.repeat_mask_4 = RepeatMask(4)
        self.out_channel_mask_4 = nn.Parameter(torch.ones(96, requires_grad=True))
        self.irb_bottleneck4 = self.irb_bottleneck(
            48, 96, 4, 2, expansion, self.out_channel_mask_4
        )

        self.repeat_mask_5 = RepeatMask(4)
        self.out_channel_mask_5 = nn.Parameter(torch.ones(128, requires_grad=True))
        self.irb_bottleneck5 = self.irb_bottleneck(
            96, 128, 4, 2, expansion, self.out_channel_mask_5
        )

        self.repeat_mask_6 = RepeatMask(3)
        self.out_channel_mask_6 = nn.Parameter(torch.ones(256, requires_grad=True))
        self.irb_bottleneck6 = self.irb_bottleneck(
            128, 256, 3, 1, expansion, self.out_channel_mask_6
        )

        self.out_channel_mask_7 = nn.Parameter(torch.ones(320, requires_grad=True))
        self.irb_bottleneck7 = self.irb_bottleneck(
            256, 320, 1, 2, expansion, self.out_channel_mask_7
        )

        # decoding arm
        self.D_irb1 = self.irb_bottleneck(
            320, 128, 1, 2, expansion, self.out_channel_mask_5, True
        )
        self.D_irb2 = self.irb_bottleneck(
            128, 96, 1, 2, expansion, self.out_channel_mask_4, True
        )
        self.D_irb3 = self.irb_bottleneck(
            96, 48, 1, 2, expansion, self.out_channel_mask_3, True
        )
        self.D_irb4 = self.irb_bottleneck(
            48, 32, 1, 2, expansion, self.out_channel_mask_2, True
        )
        self.D_irb5 = self.irb_bottleneck(
            32, 24, 1, 2, expansion, self.out_channel_mask_1, True
        )

        self.out_channel_mask_8 = nn.Parameter(torch.ones(16), requires_grad=False)
        self.D_irb6 = self.irb_bottleneck(
            24, 16, 1, 2, expansion, self.out_channel_mask_8, True
        )
        self.out_channel_mask_9 = nn.Parameter(torch.ones(3), requires_grad=False)
        self.D_irb7 = self.irb_bottleneck(
            16, 3, 1, 1, expansion, self.out_channel_mask_9, False
        )

    def depthwise_conv(self, in_c, out_c, k=3, s=1, p=0):
        conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=k, padding=p, groups=in_c, stride=s),
            nn.BatchNorm2d(num_features=in_c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=1),
        )
        return conv

    def irb_bottleneck(
        self, in_c, out_c, repeat, stride, expansion, out_channel_mask=None, up=False
    ):
        convs = []
        if up:
            xx = UpInvertedResidualBlock(
                in_c, out_c, stride, expansion, out_channel_mask
            )
            convs.append(xx)
            if repeat > 1:
                for _ in range(1, repeat):
                    xx = UpInvertedResidualBlock(
                        out_c, out_c, 1, expansion, out_channel_mask
                    )
                    convs.append(xx)
            conv = nn.Sequential(*convs)
        else:
            xx = InvertedResidualBlock(in_c, out_c, stride, expansion, out_channel_mask)
            convs.append(xx)
            if repeat > 1:
                for _ in range(1, repeat):
                    xx = InvertedResidualBlock(
                        out_c, out_c, 1, expansion, out_channel_mask
                    )
                    convs.append(xx)
            conv = nn.Sequential(*convs)
        return conv

    def irb_forward(self, blocks, x, block_mask=None):
        latency = 0
        if block_mask is None:
            for i in range(len(blocks)):
                block = blocks[i]
                x, cur_lat = block(x)
                latency += cur_lat

        else:
            repeat_mask = block_mask.forward()
            for i in range(len(blocks)):
                block = blocks[i]
                x, cur_lat = block(x)
                latency += cur_lat
                if i + 1 == repeat_mask:
                    break

        return x, latency

    def forward(self, x):
        x1 = self.conv3x3(x)  # (32, 256, 256)
        x2, lat2 = self.irb_forward(self.irb_bottleneck1, x1)  # (16,256,256) s1
        x3, lat3 = self.irb_forward(
            self.irb_bottleneck2, x2, self.repeat_mask_2
        )  # (24,128,128) s2
        x4, lat4 = self.irb_forward(
            self.irb_bottleneck3, x3, self.repeat_mask_3
        )  # (32,64,64) s3
        x5, lat5 = self.irb_forward(
            self.irb_bottleneck4, x4, self.repeat_mask_4
        )  # (64,32,32)
        x6, lat6 = self.irb_forward(
            self.irb_bottleneck5, x5, self.repeat_mask_5
        )  # (96,16,16) s4
        x7, lat7 = self.irb_forward(
            self.irb_bottleneck6, x6, self.repeat_mask_6
        )  # (160,16,16)
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



input = torch.randn(1, 3, 512, 512)
model = UNetMobileNetv3(512)
out, latency = model(input)
print(out.shape, latency.item())

um_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of model parameters: ", um_params)
