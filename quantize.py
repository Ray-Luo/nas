import os
import torch
from unet_mobilenetv3_small import UNetMobileNetv3
import torch.nn as nn
import torch.quantization

class FusedConvBNReLU(nn.Module):
    def __init__(self, conv, bn):
        super(FusedConvBNReLU, self).__init__()
        self.fused_conv = torch.nn.utils.fuse_conv_bn_relu(conv, bn)

    def forward(self, x):
        x = self.fused_conv(x)
        return x

def fuse_module(module):
    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.ReLU6)):
        return False
    return True

class FusedUNetMobileNetv3(UNetMobileNetv3):
    def __init__(self, out_size):
        super(FusedUNetMobileNetv3, self).__init__(out_size)
        for name, m in self.named_modules():
            if isinstance(m, nn.Sequential):
                layers = []
                prev_conv = None
                prev_bn = None
                for layer in m:
                    if isinstance(layer, nn.Conv2d):
                        prev_conv = layer
                    elif isinstance(layer, nn.BatchNorm2d) and prev_conv is not None:
                        prev_bn = layer
                    elif isinstance(layer, (nn.ReLU, nn.ReLU6)) and prev_conv is not None and prev_bn is not None:
                        layers.append(FusedConvBNReLU(prev_conv, prev_bn))
                        layers.append(layer)
                        prev_conv = None
                        prev_bn = None
                    elif fuse_module(layer):
                        layers.append(layer)
                if prev_conv is not None and prev_bn is not None:
                    layers.append(FusedConvBNReLU(prev_conv, prev_bn))
                else:
                    if prev_conv is not None:
                        layers.append(prev_conv)
                    if prev_bn is not None:
                        layers.append(prev_bn)
                setattr(self, name, nn.Sequential(*layers))





original_model = UNetMobileNetv3(512)
pretrained_checkpoint_path = "path/to/your/checkpoint.pth"
original_model.load_state_dict(torch.load(pretrained_checkpoint_path))

fused_model = FusedUNetMobileNetv3(512)
fused_model.load_state_dict(original_model.state_dict())
fused_model.eval()

backend = "fbgemm"  # running on a x86 CPU. Use "qnnpack" if running on ARM.

"""Insert stubs"""
fused_model = nn.Sequential(torch.quantization.QuantStub(),
    *fused_model,
    torch.quantization.DeQuantStub())

"""Prepare"""
fused_model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.quantization.prepare(fused_model, inplace=True)


"""Calibrate
- This example uses random data for convenience.
Use representative (validation) data instead.
"""
# with torch.inference_mode():
#   for _ in range(10):
#     x = torch.rand(1,2, 28, 28)
#     m(x)
