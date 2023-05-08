
import torch.nn as nn
import torch
from dynamic_model_res import UNetMobileNetv3




target = torch.randn(1,3,512,512)
input = torch.randn(1, 3, 512, 512)
model = UNetMobileNetv3(512)

# model.load_state_dict(torch.load('./my_model.pth', map_location=torch.device('cpu')))
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
