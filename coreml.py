import os
import torch
from dynamic_model_res2 import UNetMobileNetv3
from gfpgan import GFPGAN
import torch.nn as nn
import torch.quantization
import io
from torch.jit.mobile import _load_for_lite_interpreter
from contextlib import contextmanager
import cv2
from torchvision.transforms.functional import normalize

from torch.backends._coreml.preprocess import (
    CompileSpec,
    TensorSpec,
    CoreMLComputeUnit,
)

def model_spec():
    return {
        "forward": CompileSpec(
            inputs=(
                TensorSpec(
                    shape=[1, 3, 512, 512],
                ),
            ),
            outputs=(
                TensorSpec(
                    shape=[1, 3, 512, 512],
                ),
            ),
            backend=CoreMLComputeUnit.ALL,
            allow_low_precision=True,
        ),
    }


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




original_model = UNetMobileNetv3(512)
pretrained_checkpoint_path = "./last_big.ckpt"
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

output_dir = "./"
input = cv2.imread("/data/sandcastle/boxes/fbsource/fbcode/compphoto/media_quality/face_restoration/gfpgan/myself.png", cv2.IMREAD_COLOR)
input = cv2.resize(input, (512, 512))
input = img2tensor(input / 255.0, bgr2rgb=True, float32=True)
input = normalize(input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
input = input.unsqueeze(0)
print(input.shape)
output = original_model(input)
output = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
cv2.imwrite("./enhanced.png", output)

import coremltools as ct
if 0:
    onnx_filename = "coreml.onnx"
    torch.onnx.export(original_model, input, onnx_filename, input_names=["input"], output_names=["output"])

    # Load the ONNX model
    model = ct.converters.onnx.convert(onnx_filename)

    # Save the Core ML model
    coreml_filename = "./coreml.mlmodel"
    model.save(coreml_filename)

if 1:

    original_model = torch.jit.trace(original_model, input)

    original_model = ct.convert(
        original_model,
        inputs=[ct.ImageType(name="input", shape=input.shape,
        # scale=scale, bias=bias,
        color_layout=ct.colorlayout.RGB)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        outputs=[ct.ImageType(name="output",
        scale=1.0, bias=0)],
    )

    # original_model.save("model_no_metadata.mlmodel")

    # original_model = ct.models.MLModel("model_no_metadata.mlmodel")

    # original_model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "faceEnhancer"
    # import json
    # labels_json = {"labels": ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow", "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train", "tvOrMonitor"]}
    # original_model.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)


    original_model.save("model_metadata_big.mlmodel")
if 0:
    original_model = torch.jit.trace(original_model, input)
    compile_spec = model_spec()
    mlmodel = torch._C._jit_to_backend("coreml", original_model, compile_spec)
    mlmodel._save_for_lite_interpreter("./mobilenetv2_coreml.ptl")
