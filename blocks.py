import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



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


class SmartConv(nn.Module):
    def __init__(self, in_channels, out_channels, target_height, target_width, kernel_size):
        super(SmartConv, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, bias=True)

    def smart_padding(self, x, target_height, target_width):
        input_height, input_width = x.shape[2], x.shape[3]
        dilation = 1
        stride = 1
        if input_height < target_height or input_width < target_width:
            # Calculate factors for height and width, assuming square tensors
            factor_height = (target_height + input_height - 1) // input_height
            factor_width = (target_width + input_width - 1) // input_width
            assert factor_height == factor_width

            dilation = factor_height
            stride = factor_height

            # Create smart zero-padding
            x_padded = torch.zeros(x.shape[0], x.shape[1], factor_height * input_height, factor_width * input_width, device=x.device)

            x_padded[:, :, ::factor_height, ::factor_width] = x

            # Crop the padded tensor to match the target height and width
            x = x_padded[:, :, :target_height, :target_width]

        return x, dilation, stride

    def forward(self, x, padding=0):
        self.x_padding, dilation, stride = self.smart_padding(x, self.target_height, self.target_width)

        adjusted_padding = (self.kernel_size - 1) * dilation // 2 + padding

        self.conv.dilation = dilation
        self.conv.stride = stride
        self.conv.padding = adjusted_padding
        self.x_conv = self.conv(self.x_padding)

        self.out, _, _ = self.smart_padding(self.x_conv, self.target_height, self.target_width)

        return self.out


class FBNetV2BasicSearchBlock(nn.Module):
    def __init__(self, in_channels, max_out_channels, num_masks, conv_kernel_configs, subsampling_factors, target_height, target_width):
        super(FBNetV2BasicSearchBlock, self).__init__()

        self.channel_mask = ChannelMask(in_channels, max_out_channels, num_masks)

        # Initialize resolution subsampling modules and their corresponding weights
        self.resolution_subsampling_weights = nn.Parameter(torch.zeros(len(subsampling_factors)))
        self.resolution_subsampling_modules = nn.ModuleList([
            ResolutionSubsampling(factor) for factor in subsampling_factors
        ])

        # Initialize dilated convolution modules with their corresponding weights
        self.conv_kernel_weights = nn.Parameter(torch.zeros(len(conv_kernel_configs)))
        self.conv_kernel_modules = nn.ModuleList([
            SmartConv(max_out_channels, max_out_channels, target_height, target_width, kernel_size)
            for kernel_size in conv_kernel_configs
        ])

    def forward(self, x):
        # Apply channel mask
        x = self.channel_mask(x)

        # Apply resolution subsampling, smart zero-padding and dilated convolution
        x_outs = []
        for res_sub_module in self.resolution_subsampling_modules:
            x_res_sub = res_sub_module(x)

            # Apply dilated convolution (weighted sum)
            conv_weights = nn.functional.softmax(self.conv_kernel_weights, dim=0)
            x_out = sum(w * conv_module(x_res_sub) for w, conv_module in zip(conv_weights, self.conv_kernel_modules))
            x_outs.append(x_out)

        # Combine the outputs using resolution subsampling weights
        res_sub_weights = nn.functional.softmax(self.resolution_subsampling_weights, dim=0)
        x_combined = torch.stack(x_outs).permute(1, 0, 2, 3, 4)
        x_out = torch.sum(x_combined * res_sub_weights.view(1, -1, 1, 1, 1), dim=1)

        return x_out


# define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = FBNetV2BasicSearchBlock(3, max_out_channels=100, num_masks=3, conv_kernel_configs=conv_kernel_configs, subsampling_factors=subsampling_factors, target_height=32, target_width=32)
        self.conv2 = FBNetV2BasicSearchBlock(100, max_out_channels=150, num_masks=3, conv_kernel_configs=conv_kernel_configs, subsampling_factors=subsampling_factors, target_height=32, target_width=32)
        self.conv3 = FBNetV2BasicSearchBlock(150, max_out_channels=300, num_masks=3, conv_kernel_configs=conv_kernel_configs, subsampling_factors=subsampling_factors, target_height=32, target_width=32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(300 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # 100, 32, 32 --> 100, 16, 16
        x = self.pool(torch.relu(self.conv2(x))) # 150, 16, 16 --> 150, 16, 16
        x = self.pool(torch.relu(self.conv3(x))) # 150, 16, 16 --> 300, 16, 16
        x = x.view(-1, 300 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
