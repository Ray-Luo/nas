import os
import torch
from unet_mobilenetv3_small import UNetMobileNetv3
import torch.nn as nn
import torch.quantization

def fuse_conv_bn_relu(conv, bn, relu):
    # Extract Conv2d and BatchNorm2d parameters
    w = conv.weight
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias

    # Fuse parameters
    if conv.bias is None:
        b = (beta - gamma * mean / (var + eps).sqrt()).detach()
    else:
        print(conv.bias.shape, mean.shape, gamma.shape, var.shape, eps, "*****")
        b = (conv.bias - gamma * mean / (var + eps).sqrt()).detach()

    w = (w * (gamma / (var + eps).sqrt()).view(-1, 1, 1, 1)).detach()

    # Create a new fused Conv2d layer
    fused_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                           conv.dilation, conv.groups, bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)

    # Return the fused layer
    return nn.Sequential(fused_conv, nn.ReLU(inplace=True))


class FusedConvBNReLU(nn.Module):
    def __init__(self, conv, bn, relu):
        super(FusedConvBNReLU, self).__init__()
        self.fused_conv = fuse_conv_bn_relu(conv, bn, relu)

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
                prev_relu = None
                for layer in m:
                    if isinstance(layer, nn.Conv2d):
                        prev_conv = layer
                    elif isinstance(layer, nn.BatchNorm2d) and prev_conv is not None:
                        prev_bn = layer
                    elif isinstance(layer, (nn.ReLU, nn.ReLU6)) and prev_conv is not None and prev_bn is not None:
                        prev_relu = layer
                    elif fuse_module(layer):
                        if prev_conv is not None and prev_bn is not None and prev_relu is not None:
                            layers.append(FusedConvBNReLU(prev_conv, prev_bn, prev_relu))
                            prev_conv = None
                            prev_bn = None
                            prev_relu = None
                        layers.append(layer)
                if prev_conv is not None and prev_bn is not None and prev_relu is not None:
                    layers.append(FusedConvBNReLU(prev_conv, prev_bn, prev_relu))
                else:
                    if prev_conv is not None:
                        layers.append(prev_conv)
                    if prev_bn is not None:
                        layers.append(prev_bn)
                    if prev_relu is not None:
                        layers.append(prev_relu)
                setattr(self, name, nn.Sequential(*layers))




original_model = UNetMobileNetv3(512)
pretrained_checkpoint_path = "./last.ckpt"
checkpoint = torch.load(
    pretrained_checkpoint_path,
    map_location=lambda storage, loc: storage,
)["state_dict"]
filtered_checkpoint = {}
for key, value in checkpoint.items():
    target = "net_student."
    if target in key:
        filtered_checkpoint[key.replace(target, "")] = value

model_dict = original_model.state_dict()
model_dict.update(filtered_checkpoint)
original_model.load_state_dict(filtered_checkpoint)

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

"""Convert"""
torch.quantization.convert(fused_model, inplace=True)

"""Check"""
print(fused_model[[1]].weight().element_size())
