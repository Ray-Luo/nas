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
        expansion=None,
        out_channel_mask=None,
    ):
        super(InvertedResidualBase, self).__init__()
        assert stride in [1, 2]

        self.oup = oup
        hidden_dim = expansion * inp

        self.identity = stride == 1 and inp == oup

        self.out_channel_mask = out_channel_mask
        self.expansion_mask = nn.Parameter(
            torch.ones(hidden_dim, requires_grad=True), requires_grad=True
        )

        self.act_mask = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.hs = h_swish()
        self.relu = nn.ReLU(inplace=False)

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

    def get_mac(self, in_channels, out_channels, operations, input, output):
        total_macs = 0
        batch_size = 1
        if isinstance(operations, nn.Sequential):
            for operation in operations:
                if isinstance(operation, nn.Conv2d) or isinstance(operation, nn.ConvTranspose2d):
                    _, _, h_out, w_out = output.size()
                    kernel_h, kernel_w = operation.kernel_size
                    stride_h, stride_w = operation.stride
                    groups = operation.groups

                    macs_per_output_element = (in_channels // groups) * kernel_h * kernel_w
                    num_output_elements = batch_size * out_channels * h_out * w_out
                    total_macs += macs_per_output_element * num_output_elements
                    if operation.bias:
                        total_macs += out_channels * h_out * w_out

                elif isinstance(operation, nn.BatchNorm2d):
                    _, _, h_out, w_out = output.size()
                    num_output_elements = batch_size * out_channels * h_out * w_out
                    mac = 4 * num_output_elements
                    total_macs += mac

                else:
                    raise NotImplementedError(f"Operation {type(operation)} not implemented")

        elif isinstance(operations, h_swish):
            total_macs += 0

        elif isinstance(operations, nn.ReLU):
            total_macs += 0

        elif isinstance(operations, SELayer):
            _, _, h_in, w_in = input.size()

            middle_channels = _make_divisible(in_channels // 4, 8)
            linear1_macs = in_channels * middle_channels
            linear2_macs = middle_channels * out_channels

            num_elements = 1 * in_channels * h_in * w_in
            total_macs += linear1_macs + linear2_macs + num_elements

        else:
            raise NotImplementedError(f"Operation {type(operations)} not implemented")

        return torch.tensor([total_macs], dtype=torch.float, device=output.device, requires_grad=True)

    def forward(self, x):
        act_mask = torch.sigmoid(self.act_mask)
        se_mask = torch.sigmoid(self.se_mask)
        out_channel_mask = torch.sigmoid(self.out_channel_mask)
        expansion_mask = torch.sigmoid(self.expansion_mask)
        expansion_sum = torch.sum(expansion_mask)
        out_channel_sum = torch.sum(out_channel_mask)
        expansion_mask = expansion_mask.view(1, -1, 1, 1)
        out_channel_mask = out_channel_mask.view(1, -1, 1, 1)

        _, in_channels, height, width = x.size()
        if_pw = expansion_sum == in_channels
        total_macs = 0

        # point_wise conv
        if not if_pw:
            pw_out = self.pw(x)
            pw_macs = self.get_mac(in_channels, expansion_sum, self.pw, x, pw_out)  # pw

            hs_out = self.hs(pw_out)
            hs_macs = self.get_mac(expansion_sum, expansion_sum, self.hs, pw_out, hs_out)  # hs

            relu_out = self.relu(pw_out)
            relu_macs = self.get_mac(expansion_sum, expansion_sum, self.relu, pw_out, relu_out)  # relu

            out = act_mask * hs_out + (1 - act_mask) * relu_out
            total_macs += pw_macs + act_mask * hs_macs + (1 - act_mask) * relu_macs
        else:
            out = x

        # depthwise conv
        dw_out = self.dw(out)
        dw_macs = self.get_mac(expansion_sum, expansion_sum, self.dw, out, dw_out)  # pw

        hs_out = self.hs(dw_out)
        hs_macs = self.get_mac(expansion_sum, expansion_sum, self.hs, dw_out, hs_out)  # hs

        relu_out = self.relu(dw_out)
        relu_macs = self.get_mac(expansion_sum, expansion_sum, self.relu, dw_out, relu_out)  # relu

        out = act_mask * hs_out + (1 - act_mask) * relu_out
        total_macs += dw_macs + act_mask * hs_macs + (1 - act_mask) * relu_macs

        se_out = self.se(out)
        se_macs = self.get_mac(expansion_sum, expansion_sum, self.se, out, se_out)  # relu
        id_macs = 0
        out = se_out * se_mask + (1 - se_mask) * self.id(out)
        total_macs += se_mask * se_macs + (1 - se_mask) * id_macs

        # mimic skipping expansion layer
        out = out * expansion_mask

        # pointwise linear projection
        pwl_out = self.pw_linear(out)
        pwl_macs = self.get_mac(expansion_sum, out_channel_sum, self.pw_linear, out, pwl_out)  # pw
        total_macs += pwl_macs
        if self.identity:
            out = x + pwl_out

        # mimic skipping output layer
        out = pwl_out * out_channel_mask
        return out, total_macs


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
            out_channel_mask,
        )
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
            out_channel_mask,
        )
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
        self.repeat_mask_1 = 10 * nn.Parameter(torch.ones(2, requires_grad=True))
        self.out_channel_mask_1 = nn.Parameter(torch.ones(16, requires_grad=True))
        self.irb_bottleneck1 = self.irb_bottleneck(
            3, 16, 2, 2, expansion, out_channel_mask=self.out_channel_mask_1
        )

        self.repeat_mask_2 = 10 * nn.Parameter(torch.ones(2, requires_grad=True))
        self.out_channel_mask_2 = nn.Parameter(torch.ones(32, requires_grad=True))
        self.irb_bottleneck2 = self.irb_bottleneck(
            16, 32, 2, 2, expansion, self.out_channel_mask_2
        )

        self.repeat_mask_3 = 10 * nn.Parameter(torch.ones(3, requires_grad=True))
        self.out_channel_mask_3 = nn.Parameter(torch.ones(48, requires_grad=True))
        self.irb_bottleneck3 = self.irb_bottleneck(
            32, 48, 3, 2, expansion, self.out_channel_mask_3
        )

        self.repeat_mask_4 = 10 * nn.Parameter(torch.ones(4, requires_grad=True))
        self.out_channel_mask_4 = nn.Parameter(torch.ones(96, requires_grad=True))
        self.irb_bottleneck4 = self.irb_bottleneck(
            48, 96, 4, 2, expansion, self.out_channel_mask_4
        )

        self.repeat_mask_5 = 10 * nn.Parameter(torch.ones(4, requires_grad=True))
        self.out_channel_mask_5 = nn.Parameter(torch.ones(128, requires_grad=True))
        self.irb_bottleneck5 = self.irb_bottleneck(
            96, 128, 4, 2, expansion, self.out_channel_mask_5
        )

        self.repeat_mask_6 = 10 * nn.Parameter(torch.ones(3, requires_grad=True))
        self.out_channel_mask_6 = nn.Parameter(torch.ones(256, requires_grad=True))
        self.irb_bottleneck6 = self.irb_bottleneck(
            128, 256, 3, 2, expansion, self.out_channel_mask_6
        )

        self.out_channel_mask_7 = 10 * nn.Parameter(torch.ones(320, requires_grad=True))
        self.irb_bottleneck7 = self.irb_bottleneck(
            256, 320, 1, 1, expansion, self.out_channel_mask_7
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
            32, 16, 1, 2, expansion, self.out_channel_mask_1, True
        )
        self.out_channel = 10 * nn.Parameter(torch.ones(3), requires_grad=False)
        self.D_irb6 = self.irb_bottleneck(
            16, 3, 1, 2, expansion, self.out_channel, True
        )

    def depthwise_conv(self, in_c, out_c, k=3, s=1, p=0):
        conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=k, padding=p, groups=in_c, stride=s),
            nn.BatchNorm2d(num_features=in_c),
            nn.ReLU6(inplace=False),
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
        total_macs = 0
        if block_mask is None:
            for i in range(len(blocks)):
                block = blocks[i]
                x, cur_macs = block(x)
                total_macs += cur_macs

        else:
            block_mask = torch.sigmoid(block_mask)
            for i in range(len(blocks)):
                block = blocks[i]
                x, cur_macs = block(x)
                x = x * block_mask[i]
                total_macs += cur_macs * block_mask[i]

        return x, total_macs

    def forward(self, x):
        x1 = x
        x2, macs2 = self.irb_forward(self.irb_bottleneck1, x1)
        x3, macs3 = self.irb_forward(
            self.irb_bottleneck2, x2, self.repeat_mask_2
        )
        x4, macs4 = self.irb_forward(
            self.irb_bottleneck3, x3, self.repeat_mask_3
        )
        x5, macs5 = self.irb_forward(
            self.irb_bottleneck4, x4, self.repeat_mask_4
        )
        x6, macs6 = self.irb_forward(
            self.irb_bottleneck5, x5, self.repeat_mask_5
        )
        x7, macs7 = self.irb_forward(
            self.irb_bottleneck6, x6, self.repeat_mask_6
        )
        x8, macs8 = self.irb_forward(self.irb_bottleneck7, x7)

        # Right arm / Decoding arm with skip connections
        d1, macs9 = self.irb_forward(self.D_irb1, x8)
        d1 += x6
        d2, macs10 = self.irb_forward(self.D_irb2, d1)
        d2 += x5
        d3, macs11 = self.irb_forward(self.D_irb3, d2)
        d3 += x4
        d4, macs12 = self.irb_forward(self.D_irb4, d3)
        d4 += x3
        d5, macs13 = self.irb_forward(self.D_irb5, d4)
        d5 += x2
        d6, macs14 = self.irb_forward(self.D_irb6, d5)
        return (
            d6,
            macs2
            + macs3
            + macs4
            + macs5
            + macs6
            + macs7
            + macs8
            + macs9
            + macs10
            + macs11
            + macs12
            + macs13
            + macs14
        )



target = torch.randn(1,3,512,512)
input = torch.randn(1, 3, 512, 512)
model = UNetMobileNetv3(512)
out, latency_original = model(input)
print(out.shape, latency_original.item())

from thop import profile
macs, params = profile(model, inputs=(input, ))
print(macs)
# import torch.optim as optim

# optimizer = optim.SGD(model.parameters(), lr=1e-3)
# criterion = nn.L1Loss()
# num_epochs = 1000
# weight = 1e-7

# with torch.no_grad():
#     _, initial_latency = model(input)
#     target_latency = 0.5 * initial_latency.item()

# for epoch in range(num_epochs):
#     optimizer.zero_grad()

#     out, latency = model(input)
#     loss = criterion(out, target)
#     latency_constraint = torch.relu(latency - target_latency)
#     # if latency > latency_original * 0.1:
#     loss += latency_constraint * weight
#         # print(latency_original.item(), latency.item(), loss.item())

#     print("Epoch: {}, Loss: {}, Lat: {}, Ori_lat: {}".format(epoch, loss.item(), latency.item(), initial_latency.item()))
#     loss.backward()
#     # print(torch.sum(model.repeat_mask_5).item())
#     # print("******", model.irb_bottleneck2[0].expansion_mask.grad)
#     optimizer.step()

# print(torch.sum(model.irb_bottleneck2[0].expansion_mask).item())

"""
1869096064.0 1863985408.0
4287650514.0

786600.0
786483.0

1863985408.0
4287650514.0

2174149888
4287650514
"""
