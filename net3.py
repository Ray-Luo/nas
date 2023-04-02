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


class ResolutionSubsampling(nn.Module):
    def __init__(self, subsampling_factor):
        super(ResolutionSubsampling, self).__init__()
        self.subsampling_factor = subsampling_factor

    def forward(self, x):
        if self.subsampling_factor > 1:
            x = nn.functional.avg_pool2d(x, kernel_size=self.subsampling_factor, stride=self.subsampling_factor)
        return x


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


class FBNetV2BasicSearchBlock(nn.Module):
    def __init__(self, in_channels, max_out_channels, num_masks, conv_kernel_configs, subsampling_factors, target_height, target_width):
        super(FBNetV2BasicSearchBlock, self).__init__()

        self.channel_mask = ChannelMask(in_channels, max_out_channels, num_masks)

        # Initialize resolution subsampling modules and their corresponding weights
        self.resolution_subsampling_weights = nn.Parameter(torch.zeros(len(subsampling_factors)))
        self.resolution_subsampling_modules = nn.ModuleList([
            ResolutionSubsampling(factor) for factor in subsampling_factors
        ])

        self.smart_zero_padding = SmartZeroPadding()

        # Initialize dilated convolution modules with their corresponding weights
        self.conv_kernel_weights = nn.Parameter(torch.zeros(len(conv_kernel_configs)))
        self.conv_kernel_modules = nn.ModuleList([
            DilatedConvolution(max_out_channels, max_out_channels, kernel_size, stride, padding, dilation)
            for kernel_size, stride, padding, dilation in conv_kernel_configs
        ])

        self.target_height = target_height
        self.target_width = target_width

    def forward(self, x):
        # Apply channel mask
        x = self.channel_mask(x)

        # Apply resolution subsampling, smart zero-padding and dilated convolution
        x_outs = []
        for res_sub_module in self.resolution_subsampling_modules:
            x_res_sub = res_sub_module(x)

            x_padded = self.smart_zero_padding(x_res_sub, self.target_height, self.target_width)

            # Apply dilated convolution (weighted sum)
            conv_weights = nn.functional.softmax(self.conv_kernel_weights, dim=0)
            x_out = sum(w * conv_module(x_padded) for w, conv_module in zip(conv_weights, self.conv_kernel_modules))
            x_outs.append(x_out)

        # Combine the outputs using resolution subsampling weights
        res_sub_weights = nn.functional.softmax(self.resolution_subsampling_weights, dim=0)
        x_combined = torch.stack(x_outs).permute(1, 0, 2, 3, 4)
        x_out = torch.sum(x_combined * res_sub_weights.view(1, -1, 1, 1, 1), dim=1)

        return x_out



a = torch.randn(1, 3, 512, 512)

in_channels = 3
max_out_channels = 16
num_masks = 3
conv_kernel_configs = [
    [3, 1, 0, 1],
    [5, 1, 0, 1]
]
subsampling_factors = [2,4,8,16]
target_height = 512
target_width = 512

# print(c.shape)  # output: torch.Size([1, 64, 224, 224])
cm = FBNetV2BasicSearchBlock(in_channels, max_out_channels, num_masks, conv_kernel_configs, subsampling_factors, target_height, target_width)
out = cm(a)
print(out.shape)

"""
In the above implementation of channel mask, shouldn't combined_mask has the same channel dimension of x? And it
"""
