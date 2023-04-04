import torch
import torch.nn as nn
from mobilenet import InvertedResidual



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
    def __init__(self,
        in_channels,
        out_channels,
        target_height,
        target_width,
        kernel_size,
        stride,
        expansion,
        use_se=False,
        use_hs=False,
        pre_dilation=1,
        post_dilation=1):

        super(SmartConv, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion = expansion
        self.use_se = use_se
        self.use_hs = use_hs
        self.pre_dilation = pre_dilation
        self.post_dilation = post_dilation
        self.conv = InvertedResidual(
            self.in_channels,
            self.out_channels,
            self.expansion,
            self.kernel_size,
            self.stride,
            self.use_se,
            self.use_hs)


    def smart_padding(self, x):
        input_height, input_width = x.shape[2], x.shape[3]
        if input_height < self.target_height or input_width < self.target_width:
            self.use_dilation_next = True
            # Calculate factors for height and width, assuming square tensors
            factor_height = (self.target_height + input_height - 1) // input_height
            factor_width = (self.target_width + input_width - 1) // input_width
            assert factor_height == factor_width

            self.post_dilation = factor_height

            # Create smart zero-padding
            x_padded = torch.zeros(x.shape[0], x.shape[1], factor_height * input_height, factor_width * input_width, device=x.device)

            x_padded[:, :, ::factor_height, ::factor_width] = x

            # Crop the padded tensor to match the target height and width
            x = x_padded[:, :, :self.target_height, :self.target_width]

        return x, self.post_dilation

    def forward(self, x):
        x = self.conv(x)
        self.out, self.post_dilation = self.smart_padding(x)

        return self.out, self.post_dilation


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
            SmartConv(max_out_channels, max_out_channels, target_height, target_width, kernel_size, stride, expansion, use_se==1, use_hs==1)
            for kernel_size, stride, expansion, use_se, use_hs in conv_kernel_configs
        ])
        self.id_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.max_out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Apply channel mask
        x = self.channel_mask(x)

        # Apply resolution subsampling, smart zero-padding and dilated convolution
        x_outs = []
        for res_sub_module in self.resolution_subsampling_modules:
            x_res_sub = res_sub_module(x)

            # Apply dilated convolution (weighted sum)
            conv_weights = nn.functional.softmax(self.conv_kernel_weights, dim=0)
            # for w, conv_module in zip(conv_weights, self.conv_kernel_modules):
            #     tmp = conv_module(x_res_sub)
            #     print(tmp)
                # print(w, conv_module)
            x_out = sum(w * (conv_module(x_res_sub))[0] for w, conv_module in zip(conv_weights, self.conv_kernel_modules))
            x_outs.append(x_out)

        # Combine the outputs using resolution subsampling weights
        res_sub_weights = nn.functional.softmax(self.resolution_subsampling_weights, dim=0)
        x_combined = torch.stack(x_outs).permute(1, 0, 2, 3, 4)
        x_out = torch.sum(x_combined * res_sub_weights.view(1, -1, 1, 1, 1), dim=1)

        return x_out


in_channels = 3
max_out_channels = 256
num_masks = 3
# kernel_size, stride, expansion, use_se, use_hs
conv_kernel_configs = [
    [3, 1, 1, 1, 0],
    [5, 1, 1, 0, 1],
]
subsampling_factors = [1,2,4,8]
target_height = 32
target_width = 32


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
        x = self.conv1(x)
        x = self.pool(torch.relu(x)) # 100, 32, 32 --> 100, 16, 16

        # self.conv2.pre_dilation = post_dilation
        x = self.conv2(x)
        x = self.pool(torch.relu(x)) # 150, 16, 16 --> 150, 16, 16

        # self.conv3.pre_dilation = post_dilation
        x = self.conv3(x)
        x = self.pool(torch.relu(x)) # 150, 16, 16 --> 300, 16, 16
        x = x.view(-1, 300 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# net = FBNetV2BasicSearchBlock(in_channels, max_out_channels, num_masks, conv_kernel_configs, subsampling_factors, target_height, target_width)
net = Net()
input = torch.rand(1, 3, 32, 32)
out = net(input)
print(out.shape)


# target_height = 6
# target_width = 6
# in_channels = 1
# max_out_channels = 1
# kernel_size = 3
# stride=2
# expansion=2
# use_se=True
# use_hs=True
# pre_dilation=1
# post_dilation=1

# net = SmartConv(in_channels,
#     max_out_channels,
#     target_height,
#     target_width,
#     kernel_size,
#     stride,
#     expansion,
#     use_se,
#     use_hs,
#     pre_dilation,
#     post_dilation)

# input = torch.ones(1, 1, 3, 3)

# out, post_dilation = net(input)

# print(post_dilation)
# print(out.shape)
