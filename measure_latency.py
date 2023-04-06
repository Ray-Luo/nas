import torch
import torch.nn as nn
from blocks import ChannelMask, FBNetV2BasicSearchBlock
import torch.utils.benchmark as benchmark
import pandas as pd
from tqdm import tqdm

def measure_latency(operation, input_tensor, num_runs=100):
    device = torch.device('cpu')
    operation = operation.to(device)
    input_tensor = input_tensor.to(device)

    # warm up the model
    for i in range(10):
        _ = operation(input_tensor)

    timer = benchmark.Timer(
    stmt='model(input)',
    globals={'model': operation, 'input': input_tensor},
    )
    latency = timer.timeit(num_runs)

    return latency.mean * 1000


"""
measurement configs
"""
DEPTH_MAX = 150 + 1
DEPTH_MIN = 1
DEPTH_INTERVAL = 5

FEATURE_DIM_MAX = 32 + 1
FEATURE_DIM_MIN = 4
FEATURE_INTERVAL = 1

FEATURE_DEPTH_LIST = range(DEPTH_MIN, DEPTH_MAX, DEPTH_INTERVAL)
FEATURE_DIM_LIST = range(FEATURE_DIM_MIN, FEATURE_DIM_MAX, 1)


"""
Measure Channel Masking
"""
MAX_CHANNEL = 150 + 1
MAX_NUM_MASK = 5 + 1

if 1:
    operation_configs = []
    for in_channels in FEATURE_DEPTH_LIST:
        for max_channels in range(in_channels, MAX_CHANNEL, DEPTH_INTERVAL):
            if max_channels > MAX_CHANNEL:
                break
            for num_masks in range(1, MAX_NUM_MASK):
                operation_configs.append({'type': 'channel_mask', 'in_channels': in_channels, 'max_out_channels': max_channels, 'num_masks': num_masks})

    input_configs = []
    for width in FEATURE_DIM_LIST:
        input_configs.append({'type': 'tensor', 'batch_size': 1, 'height': width, 'width': width})

    latency_lut = {}
    in_channels = []
    f_height = []
    max_channels = []
    num_masks = []
    latency_mean = []

    for input_config in tqdm(input_configs):
        for operation_config in tqdm(operation_configs):
            input_tensor = torch.rand(input_config["batch_size"], operation_config['in_channels'], input_config["height"], input_config["width"])
            if operation_config['type'] == 'channel_mask':
                op = ChannelMask(operation_config['in_channels'], operation_config['max_out_channels'], operation_config['num_masks'])
                mean = measure_latency(op, input_tensor)

                in_channels.append(operation_config['in_channels'])
                f_height.append(input_config["height"])
                max_channels.append(operation_config['max_out_channels'])
                num_masks.append(operation_config['num_masks'])
                latency_mean.append(mean)

    latency_lut["in_channels"] = in_channels
    latency_lut["f_height"] = f_height
    latency_lut["max_channels"] = max_channels
    latency_lut["num_masks"] = num_masks
    latency_lut["latency_mean"] = latency_mean

    df = pd.DataFrame(latency_lut)

    # Save the DataFrame to a CSV file
    df.to_csv('/home/luoleyouluole/nas/latency_lut.csv', index=False)
