import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ChannelMask(nn.Module):
    def __init__(self, in_channels, max_out_channels, num_masks):
        super(ChannelMask, self).__init__()

        self.in_channels = in_channels
        self.max_out_channels = max_out_channels
        self.num_masks = num_masks

        self.alpha = nn.Parameter(torch.zeros(num_masks))
        self.masks = nn.Parameter(torch.rand(num_masks, max_out_channels))

    def forward(self, x):
        # Normalize alpha values using softmax
        weights = nn.functional.softmax(self.alpha, dim=0)

        # Apply STE to the masks to enforce binary values (0 or 1)
        binary_masks = self.masks.round().clamp(0, 1)

        # Compute the weighted sum of masks
        combined_mask = torch.sum(weights.view(-1, 1) * binary_masks, dim=0)

        # Make sure the input tensor has the same number of channels as max_out_channels
        if x.shape[1] != self.max_out_channels:
            batch_size = x.shape[0]
            zero_padding = torch.zeros(batch_size, self.max_out_channels - self.in_channels, x.shape[2], x.shape[3], device=x.device)
            x = torch.cat((x, zero_padding), dim=1)

        # Apply the combined mask to the input tensor
        x = x * combined_mask.view(1, self.max_out_channels, 1, 1)

        return x

    def backward(self, grad_output):
        # During the backward pass, gradients flow through the unrounded masks
        grad_input = grad_output.matmul(self.masks.view(self.max_out_channels, -1))
        return grad_input





class MixedOperation(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(MixedOperation, self).__init__()

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.alpha = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        weights = nn.functional.softmax(self.alpha, dim=0)

        x_conv3x3 = self.relu(self.bn(self.conv3x3(x)))
        x_conv5x5 = self.relu(self.bn(self.conv5x5(x)))
        x_maxpool = self.maxpool(x)

        out = weights[0] * x_conv3x3 + weights[1] * x_conv5x5 + weights[2] * x_maxpool

        return out


# class FBNetBlock(nn.Module):
#     def __init__(self, in_channels, max_out_channels, stride):
#         super(FBNetBlock, self).__init__()

#         self.channel_mask = ChannelMask(in_channels, max_out_channels)
#         self.mixed_op = MixedOperation(in_channels, max_out_channels, stride)
#         self.resolution_subsampling = ResolutionSubsampling()

#     def forward(self, x, out_channels, output_height, output_width):
#         x = self.channel_mask(x, out_channels)
#         x = self.mixed_op(x)
#         x = self.resolution_subsampling(x, output_height, output_width)

#         return x


# class FBNetV2Example(nn.Module):
#     def __init__(self, num_classes=10):
#         super(FBNetV2Example, self).__init__()

#         self.block1 = FBNetBlock(3, 64, stride=1)
#         self.block2 = FBNetBlock(64, 128, stride=2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x, block1_channels, block2_channels, block2_height, block2_width):
#         x = self.block1(x, block1_channels, x.shape[2], x.shape[3])
#         x = self.block2(x, block2_channels, block2_height, block2_width)
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x


class SmartZeroPadding(nn.Module):
    def __init__(self):
        super(SmartZeroPadding, self).__init__()

    def forward(self, x, target_height, target_width):
        input_height, input_width = x.shape[2], x.shape[3]

        if input_height < target_height or input_width < target_width:
            # Calculate factors for height and width
            factor_height = (target_height + input_height - 1) // input_height
            factor_width = (target_width + input_width - 1) // input_width

            # Create smart zero-padding
            x_padded = torch.zeros(x.shape[0], x.shape[1], factor_height * input_height, factor_width * input_width, device=x.device)

            x_padded[:, :, ::factor_height, ::factor_width] = x

            # Crop the padded tensor to match the target height and width
            x = x_padded[:, :, :target_height, :target_width]

        return x


class DilatedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(DilatedConvolution, self).__init__()

        # Calculate the padding required to keep the same output size
        adjusted_padding = (kernel_size - 1) * dilation // 2 + padding

        # Create a dilated convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, adjusted_padding, dilation=dilation, bias=False)

    def forward(self, x):
        return self.conv(x)


a = torch.randn(5, 16, 224, 224)
# a = torch.randn(1, 1, 4, 4)
# net = SmartZeroPadding()
# net(a, 8, 8)

# a = a.reshape(1, -1, 16)

# create tensor b with appropriate shape
# b = torch.randn(16, 64)

# perform matrix multiplication of a and b
# c = torch.matmul(a, b).reshape(1, 64, 224, 224)

# print(c.shape)  # output: torch.Size([1, 64, 224, 224])
cm = ChannelMask(16, 16, 3)
out = cm(a)

"""
In the above implementation of channel mask, shouldn't combined_mask has the same channel dimension of x? And it
"""
