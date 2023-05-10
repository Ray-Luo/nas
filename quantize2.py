import os
import torch
from unet_mobilenetv3_small import InvertedResidualBlock, UNetMobileNetv3, UpInvertedResidualBlock
import torch.nn as nn
import torch.quantization
import io
from torch.jit.mobile import _load_for_lite_interpreter
from contextlib import contextmanager
import cv2
from torchvision.transforms.functional import normalize

def tensor2img(tensor, rgb2bgr=True, min_max=(0, 1)):
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

@contextmanager
def override_quantized_engine(qengine):
    previous = torch.backends.quantized.engine
    torch.backends.quantized.engine = qengine
    try:
        yield
    finally:
        torch.backends.quantized.engine = previous

def compare_script_and_mobile(model: torch.nn.Module,
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
        if isinstance(m, InvertedResidualBlock) or isinstance(m, UpInvertedResidualBlock):
            torch.quantization.fuse_modules(m.conv, ['0', '1', '2'], inplace=True)
            if len(m.conv) < 7:
                torch.quantization.fuse_modules(m.conv, ['4', '5'], inplace=True)
            else:
                torch.quantization.fuse_modules(m.conv, ['3', '4', '6'], inplace=True)
                torch.quantization.fuse_modules(m.conv, ['7', '8'], inplace=True)

    return model


original_model = UNetMobileNetv3(512)
print(original_model)
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
# for name, module in quantized_model.named_modules():
#     if isinstance(module, nn.quantized.Conv2d) or isinstance(module, nn.quantized.Linear):
#         print(f"{name}: {module.weight().dtype}")

output_dir = "./"
input = cv2.imread("/data/sandcastle/boxes/fbsource/fbcode/compphoto/media_quality/face_restoration/gfpgan/test/1.png", cv2.IMREAD_COLOR)
cv2.resize(input, (512, 512))
input = img2tensor(input / 255.0, bgr2rgb=True, float32=True)
normalize(input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
input = input.unsqueeze(0)
print(input.shape)


output = quantized_model(input)
output = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1)).astype("uint8")
cv2.imwrite("./enhanced.png", output)

# compare_script_and_mobile(model=quantized_model, input=input)

mapping_net_trace = torch.jit.trace(quantized_model, input)
torch.jit.save(mapping_net_trace, os.path.join(output_dir, "face_res.pt"))
m_script = torch.jit.load(os.path.join(output_dir, "face_res.pt"))
ops = torch.jit.export_opnames(m_script)
print(ops)
m_script._save_for_lite_interpreter(
    os.path.join(output_dir, "latest.ptl")
)
