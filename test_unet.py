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
    def __init__(
        self,
        inp,
        oup,
        stride,
        expansion_mask=None,
        out_channel_mask=None,
        use_se=False,
        use_hs=False,
    ):
        super(InvertedResidualBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = 6 * inp

        self.identity = stride == 1 and inp == oup

        self.out_channel_mask = out_channel_mask
        self.expansion_mask = expansion_mask # nn.Parameter(torch.rand(hidden_dim))

        self.act_mask = torch.Tensor([0])
        self.hs = h_swish()
        self.relu = nn.ReLU(inplace=True)

        self.pw = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
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
        self.se = SELayer(hidden_dim) if use_se else nn.Identity()
        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        act_mask = torch.round(torch.sigmoid(self.act_mask))
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
        else:
            out = x

        # depthwise conv
        out = self.dw(out)
        out = act_mask * self.hs(out) + (1 - act_mask) * self.relu(out)
        out = self.se(out)
        out = out * expansion_mask
        # pointwise linear projection
        out = self.pw_linear(out)

        if self.identity:
            out = x + out

        out = out * out_channel_mask
        return out


class UpInvertedResidualBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expansion,
        out_channel_mask=None,
        use_se=False,
        use_hs=False,
    ):
        super(UpInvertedResidualBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = expansion * inp

        self.identity = stride == 1 and inp == oup

        self.out_channel_mask = out_channel_mask
        self.expansion_mask = nn.Parameter(torch.rand(hidden_dim))

        self.act_mask = nn.Parameter(torch.rand(1), requires_grad=True)
        self.hs = h_swish()
        self.relu = nn.ReLU(inplace=True)

        self.pw = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
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
        self.se = SELayer(hidden_dim) if use_se else nn.Identity()
        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        act_mask = torch.round(torch.sigmoid(self.act_mask))
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
        else:
            out = x

        # depthwise conv
        out = self.dw(out)
        out = act_mask * self.hs(out) + (1 - act_mask) * self.relu(out)
        out = self.se(out)
        out = out * expansion_mask
        # pointwise linear projection
        out = self.pw_linear(out)

        if self.identity:
            out = x + out

        out = out * out_channel_mask
        return out

class InvertedResidualBlock2(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expansion_mask=None,
        out_channel_mask=None,
        use_se=False,
        use_hs=False,
    ):
        super(InvertedResidualBlock2, self).__init__()
        assert stride in [1, 2]

        hidden_dim = 6 * inp

        self.identity = stride == 1 and inp == oup

        self.out_channel_mask = out_channel_mask
        self.expansion_mask = expansion_mask

        self.act_mask = torch.Tensor([0])
        self.hs = h_swish()
        self.relu = nn.ReLU(inplace=True)

        self.pw = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
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
        self.se = SELayer(hidden_dim) if use_se else nn.Identity()
        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward_gt(self, x):
        act_mask = torch.round(torch.sigmoid(self.act_mask))
        if_pw = False

        # point_wise conv
        if not if_pw:
            out = self.pw(x)
            out = act_mask * self.hs(out) + (1 - act_mask) * self.relu(out)
            # print(out[:,:-3,:,:].shape, torch.sum(out[:,:-3,:,:]).item(),"*******")
        else:
            out = x

        # depthwise conv
        out = self.dw(out)
        out = act_mask * self.hs(out) + (1 - act_mask) * self.relu(out)
        out = self.se(out)
        print(out[:,:-3,:,:].shape, torch.sum(out[:,:-3,:,:]).item(),"**** 1 ***")
        # pointwise linear projection
        out = self.pw_linear(out * expansion_mask)
        print(out.shape, torch.sum(out).item(), "**** 2 ***")

        if self.identity:
            out = x + out

        print(out[:,:-1,:,:].shape, torch.sum(out[:,:-1,:,:]).item(),"**** 3 ***")
        return out

    def forward_test(self, x):
        act_mask = torch.round(torch.sigmoid(self.act_mask))
        # out_channel_mask = torch.round(torch.sigmoid(self.out_channel_mask)).view(
        #     1, -1, 1, 1
        # )
        out_channel_mask = self.out_channel_mask
        expansion_mask = self.expansion_mask
        # expansion_mask = torch.round(torch.sigmoid(self.expansion_mask)).view(
        #     1, -1, 1, 1
        # )
        if_pw = torch.sum(expansion_mask) == x.shape[1]

        # point_wise conv
        if not if_pw:
            out = self.pw(x)
            out = act_mask * self.hs(out) + (1 - act_mask) * self.relu(out)
        else:
            out = x

        # depthwise conv
        out = self.dw(out)
        out = act_mask * self.hs(out) + (1 - act_mask) * self.relu(out)
        out = self.se(out)
        out = out * expansion_mask
        # pointwise linear projection
        out = self.pw_linear(out)

        if self.identity:
            out = x + out

        out = out * out_channel_mask
        return out


"""
1. expansion with 6 and 1 zero is equivalent to with 5
2. output with 6 and 1 zero is equivalent to with 5
"""

# input = torch.randn(1, 3, 512, 512)
# out_channel_mask = torch.Tensor([1,1,1,1,1,1,1,1,1,0]).view(1, -1, 1, 1)
# expansion = [1] * 3 * 6
# expansion[-3:] = [0,0,0]
# expansion_mask = torch.Tensor(expansion).view(1, -1, 1, 1)
# m1 = InvertedResidualBlock2(
#         3,
#         10,
#         1,
#         expansion_mask=expansion_mask,
#         out_channel_mask=out_channel_mask,
#     )
# out1 = m1.forward_gt(input)
# print()
# out2 = m1.forward_test(input)

import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSoftmaxOneHot(nn.Module):
    def __init__(self, num_classes):
        super(GumbelSoftmaxOneHot, self).__init__()
        self.p = nn.Parameter(torch.randn(num_classes), requires_grad=True)
        self.temperature = 1.0

    def forward(self, hard=True):
        logits = F.gumbel_softmax(self.p, tau=self.temperature, hard=hard)
        one_hot_index = torch.argmax(logits)
        return one_hot_index

num_classes = 5
model = GumbelSoftmaxOneHot(num_classes)
out = model()
print(out + 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# print(torch.sum(out1).item(), torch.sum(out2).item())

# import nni.retiarii.nn.pytorch as nn
# from nn_meter import load_latency_predictor
# from lut.predictor import load_latency_predictor

# input_shape = (1, 3, 512, 512)

# predictor = load_latency_predictor("cortexA76cpu_tflite21", 1.0) # case insensitive in backend
# lat = predictor.predict(net, "torch", input_shape=input_shape, apply_nni=False)

# print(lat)
