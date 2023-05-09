import os
import torch
from unet_mobilenetv3_small import InvertedResidualBlock, UNetMobileNetv3, UpInvertedResidualBlock
import torch.nn as nn
import torch.quantization

def _compare_script_and_mobile(model: torch.nn.Module,
                                input: torch.Tensor):
    # Compares the numerical outputs for script and lite modules
    qengine = "qnnpack"
    with override_quantized_engine(qengine):
        script_module = torch.jit.script(model)
        script_module_result = script_module(input)

        max_retry = 5
        for retry in range(1, max_retry + 1):
            # retries `max_retry` times; breaks iff succeeds else throws exception
            try:
                buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
                buffer.seek(0)
                mobile_module = _load_for_lite_interpreter(buffer)

                mobile_module_result = mobile_module(input)

                torch.testing.assert_close(script_module_result, mobile_module_result)
                mobile_module_forward_result = mobile_module.forward(input)
                torch.testing.assert_close(script_module_result, mobile_module_forward_result)

                mobile_module_run_method_result = mobile_module.run_method("forward", input)
                torch.testing.assert_close(script_module_result, mobile_module_run_method_result)
            except AssertionError as e:
                if retry == max_retry:
                    raise e
                else:
                    continue
            break

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

"""Prepare"""
original_model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
fused_model = fuse_model(original_model)
prepared_model = torch.quantization.prepare(fused_model)


"""Calibrate
- This example uses random data for convenience.
Use representative (validation) data instead.
"""
# with torch.inference_mode():
#   for _ in range(10):
#     x = torch.rand(1,2, 28, 28)
#     m(x)

"""Convert"""
quantized_model = torch.quantization.convert(prepared_model)

"""Check"""
for name, module in quantized_model.named_modules():
    if isinstance(module, nn.quantized.Conv2d) or isinstance(module, nn.quantized.Linear):
        print(f"{name}: {module.weight().dtype}")

output_dir = "./"
input = torch.rand(1, 3, 512, 512)
quantized_input = torch.quantize_per_tensor(input, scale=quantized_model.quant.scale, zero_point=quantized_model.quant.zero_point, dtype=torch.quint8)


out = quantized_model(input)
print(out.shape)
mapping_net_trace = torch.jit.trace(quantized_model, input)
torch.jit.save(mapping_net_trace, os.path.join(output_dir, "face_res.pt"))
m_script = torch.jit.load(os.path.join(output_dir, "face_res.pt"))
ops = torch.jit.export_opnames(m_script)
print(ops)
m_script._save_for_lite_interpreter(
    os.path.join(output_dir, "latest.ptl")
)
