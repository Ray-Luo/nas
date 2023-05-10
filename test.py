import torch
import torch.nn as nn

# Define the alternative layers
upsample = nn.Upsample(scale_factor=2, mode='bilinear')  # You can use other modes like 'bilinear' as well
conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

# Example input tensor
input_tensor = torch.randn(1, 64, 32, 32)

# Apply the layers
output = upsample(input_tensor)
output = conv(output)
print(output.shape)
