import os
import torch
from unet_mobilenetv3_small import InvertedResidualBlock, UNetMobileNetv3, UpInvertedResidualBlock
import torch.nn as nn
import torch.quantization

def fuse_model(model):
    torch.quantization.fuse_modules(model.conv3x3, ['0', '1'], inplace=True)

    for m in model.modules():
        if isinstance(m, InvertedResidualBlock):
            torch.quantization.fuse_modules(m.conv, ['0', '1', '2'], inplace=True)
            if len(m.conv) < 7:
                torch.quantization.fuse_modules(m.conv, ['4', '5'], inplace=True)
            else:
                torch.quantization.fuse_modules(m.conv, ['3', '4', '6'], inplace=True)
                torch.quantization.fuse_modules(m.conv, ['7', '8'], inplace=True)

    return model


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
original_model = original_model.cpu()
original_model.eval()

fused_model = fuse_model(original_model)


backend = "qnnpack"  # running on a x86 CPU. Use "qnnpack" if running on ARM.

"""Insert stubs"""
fused_model_modules = list(fused_model.children())
fused_model_modules.insert(0, torch.quantization.QuantStub())
fused_model_modules.append(torch.quantization.DeQuantStub())
fused_model = nn.Sequential(*fused_model_modules)


"""Prepare"""
from torch.quantization import default_qconfig

# Set the qconfig to use per-tensor observers for weights
fused_model.qconfig = default_qconfig

# Prepare the model for quantization
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
quantized_model = torch.quantization.convert(fused_model, inplace=True)

"""Check"""
for name, module in quantized_model.named_modules():
    if isinstance(module, nn.quantized.Conv2d) or isinstance(module, nn.quantized.Linear):
        print(f"{name}: {module.weight().dtype}")

output_dir = "./"
input = torch.rand(1, 3, 512, 512)

out = quantized_model(input)
print(out.shape)
mapping_net_trace = torch.jit.trace(quantized_model, input)
torch.jit.save(mapping_net_trace, os.path.join(output_dir, "face_res.pt"))
m_script = torch.jit.load(os.path.join(output_dir, "face_res.pt"))
ops = torch.jit.export_opnames(m_script)
print(ops)
m_script._save_for_lite_interpreter(
    os.path.join(output_dir, "face_res_mobileunetv2_lite.pt")
)
