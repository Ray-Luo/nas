import torch
import torch.nn as nn

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


def _make_divisible(v, divisor, min_value=None):
    # ensure that all layers have a channel number that is divisible by 8

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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

def get_mac(in_channels, out_channels, operations, input, output):
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

input = torch.randn(1, 3, 512, 512)

inp = 3
hidden_dim = 3 * 6
pw = nn.Sequential(
    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
    nn.BatchNorm2d(hidden_dim),
)

# model = h_swish()
# model = nn.ReLU()
model = SELayer(3)
# model = nn.Linear(3, 8)
# avg_pool = nn.AdaptiveAvgPool2d(1)
# input = avg_pool(input).view(1,3)
out = model(input)
macs = get_mac(inp, hidden_dim, model, input, out)
print(macs.item())

from thop import profile
macs, params = profile(model, inputs=(input, ))
print(macs)
