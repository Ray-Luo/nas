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

        self.act_mask = nn.Parameter(torch.ones(1, requires_grad=True))
        self.hs = h_swish()
        self.relu = nn.ReLU(inplace=False)

        self.se_mask = nn.Parameter(torch.ones(1, requires_grad=True))
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
                if isinstance(operation, nn.Conv2d) or isinstance(
                    operation, nn.ConvTranspose2d
                ):
                    _, _, h_out, w_out = output.size()
                    kernel_h, kernel_w = operation.kernel_size
                    stride_h, stride_w = operation.stride
                    groups = operation.groups

                    macs_per_output_element = (
                        (in_channels // groups) * kernel_h * kernel_w
                    )
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
                    raise NotImplementedError(
                        f"Operation {type(operation)} not implemented"
                    )

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

        return torch.tensor(
            total_macs, dtype=torch.float, device=output.device, requires_grad=True
        )

    def forward(self, x):
        act_mask = nn.functional.relu(self.act_mask)
        se_mask = nn.functional.relu(self.se_mask)
        out_channel_mask = nn.functional.relu(self.out_channel_mask)
        expansion_mask = nn.functional.relu(self.expansion_mask)
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
            hs_macs = self.get_mac(
                expansion_sum, expansion_sum, self.hs, pw_out, hs_out
            )  # hs

            relu_out = self.relu(pw_out)
            relu_macs = self.get_mac(
                expansion_sum, expansion_sum, self.relu, pw_out, relu_out
            )  # relu

            out = act_mask * hs_out + (1 - act_mask) * relu_out
            total_macs += pw_macs + act_mask * hs_macs + (1 - act_mask) * relu_macs
        else:
            out = x

        # depthwise conv
        dw_out = self.dw(out)
        dw_macs = self.get_mac(expansion_sum, expansion_sum, self.dw, out, dw_out)  # pw

        hs_out = self.hs(dw_out)
        hs_macs = self.get_mac(
            expansion_sum, expansion_sum, self.hs, dw_out, hs_out
        )  # hs

        relu_out = self.relu(dw_out)
        relu_macs = self.get_mac(
            expansion_sum, expansion_sum, self.relu, dw_out, relu_out
        )  # relu

        out = act_mask * hs_out + (1 - act_mask) * relu_out
        total_macs += dw_macs + act_mask * hs_macs + (1 - act_mask) * relu_macs

        se_out = self.se(out)
        se_macs = self.get_mac(
            expansion_sum, expansion_sum, self.se, out, se_out
        )  # relu
        id_macs = 0
        out = se_out * se_mask + (1 - se_mask) * self.id(out)
        total_macs += se_mask * se_macs + (1 - se_mask) * id_macs

        # mimic skipping expansion layer
        out = out * expansion_mask

        # pointwise linear projection
        pwl_out = self.pw_linear(out)
        pwl_macs = self.get_mac(
            expansion_sum, out_channel_sum, self.pw_linear, out, pwl_out
        )  # pw
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

        repeat_masks = [0, 1, 1, 2, 3, 3, 2]

        # encoding arm
        self.repeat_mask_1 = nn.Parameter(
            torch.ones(repeat_masks[1], requires_grad=True)
        )
        self.out_channel_mask_1 = nn.Parameter(torch.ones(16, requires_grad=True))
        self.db1 = self.irb_bottleneck(
            3, 16, repeat_masks[1], 2, expansion, self.out_channel_mask_1
        )

        self.repeat_mask_2 = nn.Parameter(
            torch.ones(repeat_masks[2], requires_grad=True)
        )
        self.out_channel_mask_2 = nn.Parameter(torch.ones(32, requires_grad=True))
        self.db2 = self.irb_bottleneck(
            16, 32, repeat_masks[2], 2, expansion, self.out_channel_mask_2
        )

        self.repeat_mask_3 = nn.Parameter(
            torch.ones(repeat_masks[3], requires_grad=True)
        )
        self.out_channel_mask_3 = nn.Parameter(torch.ones(48, requires_grad=True))
        self.db3 = self.irb_bottleneck(
            32, 48, repeat_masks[3], 2, expansion, self.out_channel_mask_3
        )

        self.repeat_mask_4 = nn.Parameter(
            torch.ones(repeat_masks[4], requires_grad=True)
        )
        self.out_channel_mask_4 = nn.Parameter(torch.ones(96, requires_grad=True))
        self.db4 = self.irb_bottleneck(
            48, 96, repeat_masks[4], 2, expansion, self.out_channel_mask_4
        )

        self.repeat_mask_5 = nn.Parameter(
            torch.ones(repeat_masks[5], requires_grad=True)
        )
        self.out_channel_mask_5 = nn.Parameter(torch.ones(128, requires_grad=True))
        self.db5 = self.irb_bottleneck(
            96, 128, repeat_masks[5], 2, expansion, self.out_channel_mask_5
        )

        self.repeat_mask_6 = nn.Parameter(
            torch.ones(repeat_masks[6], requires_grad=True)
        )
        self.out_channel_mask_6 = nn.Parameter(torch.ones(256, requires_grad=True))
        self.db6 = self.irb_bottleneck(
            128, 256, repeat_masks[6], 2, expansion, self.out_channel_mask_6
        )

        self.out_channel_mask_7 = nn.Parameter(torch.ones(320, requires_grad=True))
        self.db7 = self.irb_bottleneck(
            256, 320, 0, 1, expansion, self.out_channel_mask_7
        )

        # decoding arm
        self.ub1 = self.irb_bottleneck(
            320, 128, 0, 2, expansion, self.out_channel_mask_5, True
        )
        self.ub2 = self.irb_bottleneck(
            128, 96, 0, 2, expansion, self.out_channel_mask_4, True
        )
        self.ub3 = self.irb_bottleneck(
            96, 48, 0, 2, expansion, self.out_channel_mask_3, True
        )
        self.ub4 = self.irb_bottleneck(
            48, 32, 0, 2, expansion, self.out_channel_mask_2, True
        )
        self.ub5 = self.irb_bottleneck(
            32, 16, 0, 2, expansion, self.out_channel_mask_1, True
        )
        self.out_channel = nn.Parameter(torch.ones(3), requires_grad=False)
        self.ub6 = self.irb_bottleneck(16, 3, 0, 2, expansion, self.out_channel, True)

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

    def irb_bottleneck(
        self, in_c, out_c, repeat, stride, expansion, out_channel_mask=None, up=False
    ):
        convs = []
        if up:
            xx = UpInvertedResidualBlock(
                in_c, out_c, stride, expansion, out_channel_mask
            )
            convs.append(xx)
            for _ in range(repeat):
                xx = UpInvertedResidualBlock(
                    out_c, out_c, 1, expansion, out_channel_mask
                )
                convs.append(xx)
            conv = nn.Sequential(*convs)
        else:
            xx = InvertedResidualBlock(in_c, out_c, stride, expansion, out_channel_mask)
            convs.append(xx)

            for _ in range(repeat):
                xx = InvertedResidualBlock(out_c, out_c, 1, expansion, out_channel_mask)
                convs.append(xx)
            conv = nn.Sequential(*convs)
        return conv

    def irb_forward(self, blocks, x, block_mask=None):
        # at least one block
        x, total_macs = blocks[0](x)

        if block_mask is None:
            for i in range(1, len(blocks)):
                block = blocks[i]
                x, cur_macs = block(x)
                total_macs += cur_macs

        else:
            assert len(block_mask) == len(blocks) - 1
            block_mask = nn.functional.relu(block_mask)
            for i in range(1, len(blocks)):
                block = blocks[i]
                x, cur_macs = block(x)
                x = x * block_mask[i - 1]
                total_macs += cur_macs * block_mask[i - 1]

        return x, total_macs

    def disable_architecture_search(self):
        out_channel_masks = [
            self.out_channel_mask_1,
            self.out_channel_mask_2,
            self.out_channel_mask_3,
            self.out_channel_mask_4,
            self.out_channel_mask_5,
            self.out_channel_mask_6,
            self.out_channel_mask_7,
            self.out_channel,
        ]
        blocks = [
            self.db1,
            self.db2,
            self.db3,
            self.db4,
            self.db5,
            self.db6,
            self.db7,
            self.ub1,
            self.ub2,
            self.ub3,
            self.ub4,
            self.ub5,
            self.ub6,
        ]
        repeat_masks = [
            self.repeat_mask_1,
            self.repeat_mask_2,
            self.repeat_mask_3,
            self.repeat_mask_4,
            self.repeat_mask_5,
            self.repeat_mask_6,
        ]
        for item in out_channel_masks:
            item.requires_grad = False

        for item in repeat_masks:
            item.requires_grad = False

        for item in blocks:
            item.expansion_mask.requires_grad = False
            item.act_mask.requires_grad = False
            item.se_mask.requires_grad = False

    def forward(self, x):
        x1 = x
        x2, macs2 = self.irb_forward(self.db1, x1, self.repeat_mask_1)
        x3, macs3 = self.irb_forward(self.db2, x2, self.repeat_mask_2)
        x4, macs4 = self.irb_forward(self.db3, x3, self.repeat_mask_3)
        x5, macs5 = self.irb_forward(self.db4, x4, self.repeat_mask_4)
        x6, macs6 = self.irb_forward(self.db5, x5, self.repeat_mask_5)
        x7, macs7 = self.irb_forward(self.db6, x6, self.repeat_mask_6)
        x8, macs8 = self.irb_forward(self.db7, x7)

        # Right arm / Decoding arm with skip connections
        d1, macs9 = self.irb_forward(self.ub1, x8)
        d1 += x6
        d2, macs10 = self.irb_forward(self.ub2, d1)
        d2 += x5
        d3, macs11 = self.irb_forward(self.ub3, d2)
        d3 += x4
        d4, macs12 = self.irb_forward(self.ub4, d3)
        d4 += x3
        d5, macs13 = self.irb_forward(self.ub5, d4)
        d5 += x2
        d6, macs14 = self.irb_forward(self.ub6, d5)
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
            + macs14,
        )


if 0:

    target = torch.randn(1,3,512,512).cuda()
    input = torch.randn(1, 3, 512, 512).cuda()
    model = UNetMobileNetv3(512).cuda()

    import torch.optim as optim

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    num_epochs = 1000
    weight = 1e-2

    with torch.no_grad():
        _, initial_latency = model(input)
        target_latency = 0.5 * initial_latency.item()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        out, latency = model(input)
        loss = criterion(out, target)
        latency_constraint = torch.mean(torch.relu((latency - target_latency) / initial_latency))
        loss += latency_constraint * weight
        if latency_constraint == 0:
            model.disable_architecture_search()

        print("Epoch: {}, Loss: {}, LC: {}, Lat: {}, Ori_lat: {}".format(epoch, loss.item(), latency_constraint.item(), latency.item(), initial_latency.item()))
        loss.backward()
    #     # print(torch.sum(model.repeat_mask_5).item())
    #     # print("******", model.irb_bottleneck2[0].expansion_mask.grad)
        optimizer.step()

    # print(torch.sum(model.irb_bottleneck2[0].expansion_mask).item())
    torch.save(model.state_dict(), './my_model.pth')
"""
baseline_512 --> 327401472.0 --> 70 ms
baseline_256 -->  49278976.0 --> 40 ms
"""
