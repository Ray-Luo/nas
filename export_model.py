
import torch.nn as nn
import torch
from dynamic_model_res2 import UNetMobileNetv3




target = torch.randn(1,3,512,512)
input = torch.randn(1, 3, 512, 512)
model = UNetMobileNetv3(512)

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

model_dict = model.state_dict()
model_dict.update(filtered_checkpoint)
model.load_state_dict(filtered_checkpoint)
model = model.cpu()
model.eval()


import os
def to_onnx(model, input, output_dir):

    print("convert synthesis network...")
    mapping_net_trace = torch.jit.trace(model, input)
    torch.jit.save(mapping_net_trace, os.path.join(output_dir, "face_res.pt"))
    m_script = torch.jit.load(os.path.join(output_dir, "face_res.pt"), map_location=torch.device('cpu'))
    ops = torch.jit.export_opnames(m_script)
    print(ops)
    m_script._save_for_lite_interpreter(
        os.path.join(output_dir, "latest.ptl")
    )

to_onnx(model, input,  "./")
