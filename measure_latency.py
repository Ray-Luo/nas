import torch
import torch.nn as nn
from blocks import ChannelMask, ResolutionSubsampling, FBNetV2BasicSearchBlock
from mobilenet import InvertedResidual
import torch.utils.benchmark as benchmark
import pandas as pd
from tqdm import tqdm

def measure_latency(operation, input_tensor, num_runs=20):
    device = torch.device('cpu')
    operation = operation.to(device)
    input_tensor = input_tensor.to(device)

    # warm up the model
    for i in range(5):
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
FEATURE_INTERVAL = 2

FEATURE_DEPTH_LIST = range(DEPTH_MIN, DEPTH_MAX, DEPTH_INTERVAL)
FEATURE_DIM_LIST = range(FEATURE_DIM_MIN, FEATURE_DIM_MAX, 1)


"""
Measure Channel Masking
"""
MAX_CHANNEL = 150 + 1
MAX_NUM_MASK = 5 + 1

if 0:
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


"""
Measure Subsampling
"""
SUBSAMPLE_LIST = [1,2]

DEPTH_MAX = 150 + 1
DEPTH_MIN = 1
DEPTH_INTERVAL = 5

CHANNEL_LIST = range(DEPTH_MIN, DEPTH_MAX, DEPTH_INTERVAL)

if 0:
    operation_configs = []
    for subsample in SUBSAMPLE_LIST:
        operation_configs.append({'type': 'ResolutionSubsampling', 'subsampling_factor': subsample})

    input_configs = []
    for width in FEATURE_DIM_LIST:
        for in_channel in CHANNEL_LIST:
            input_configs.append({'type': 'tensor', 'batch_size': 1, "in_channels": in_channel, 'height': width, 'width': width})

    latency_lut = {}
    in_channels = []
    f_height = []
    subsampling_factor = []
    latency_mean = []

    for input_config in tqdm(input_configs):
        for operation_config in tqdm(operation_configs):
            input_tensor = torch.rand(input_config["batch_size"], input_config['in_channels'], input_config["height"], input_config["width"])
            if operation_config['type'] == 'ResolutionSubsampling':
                op = ResolutionSubsampling(operation_config['subsampling_factor'])
                mean = measure_latency(op, input_tensor)
                in_channels.append(input_config['in_channels'])
                f_height.append(input_config["height"])
                subsampling_factor.append(operation_config['subsampling_factor'])
                latency_mean.append(mean)

    latency_lut["in_channels"] = in_channels
    latency_lut["f_height"] = f_height
    latency_lut["subsampling_factor"] = subsampling_factor
    latency_lut["latency_mean"] = latency_mean

    df = pd.DataFrame(latency_lut)

    # Save the DataFrame to a CSV file
    df.to_csv('/home/luoleyouluole/nas/latency_lut.csv', index=False)


"""
Measure InvertedResidual
"""
SUBSAMPLE_LIST = [1,2]

DEPTH_MAX = 150 + 1
DEPTH_MIN = 1
DEPTH_INTERVAL = 10

CHANNEL_LIST = range(DEPTH_MIN, DEPTH_MAX, DEPTH_INTERVAL)

if 1:
    operation_configs = []
    for use_se in [0, 1]:
        for use_hs in [0, 1]:
            for kernel_size in [3, 5]:
                for in_channel in CHANNEL_LIST:
                    for out_channel in range(in_channel+1, DEPTH_MAX, DEPTH_INTERVAL):
                        operation_configs.append({'type': 'InvertedResidual', 'in_channel': in_channel, 'out_channel': out_channel, 'expansion': 1, 'kernel_size': kernel_size, 'stride': 1, 'use_se': use_se, 'use_hs': use_hs})

    input_configs = []
    for width in FEATURE_DIM_LIST:
        for in_channel in CHANNEL_LIST:
            input_configs.append({'type': 'tensor', 'batch_size': 1, "in_channels": in_channel, 'height': width, 'width': width})

    latency_lut = {}
    in_channels = []
    out_channels = []
    f_height = []
    kernel_size = []
    use_se = []
    use_hs = []
    latency_mean = []

    for input_config in tqdm(input_configs):
        for operation_config in tqdm(operation_configs):
            input_tensor = torch.rand(input_config["batch_size"], input_config['in_channels'], input_config["height"], input_config["width"])

            op = InvertedResidual(input_config['in_channels'], operation_config['out_channel'], operation_config['expansion'], operation_config['kernel_size'], operation_config['stride'], operation_config['use_se'],operation_config['use_hs'])

            mean = measure_latency(op, input_tensor)

            in_channels.append(input_config['in_channels'])
            f_height.append(input_config["height"])
            out_channels.append(operation_config['out_channel'])
            kernel_size.append(operation_config['kernel_size'])
            use_se.append(operation_config['use_se'])
            use_hs.append(operation_config['use_hs'])
            latency_mean.append(mean)

    latency_lut["in_channels"] = in_channels
    latency_lut["f_height"] = f_height
    latency_lut["kernel_size"] = kernel_size
    latency_lut["use_se"] = use_se
    latency_lut["use_hs"] = use_hs
    latency_lut["out_channels"] = out_channels
    latency_lut["latency_mean"] = latency_mean

    df = pd.DataFrame(latency_lut)

    # Save the DataFrame to a CSV file
    df.to_csv('/home/luoleyouluole/nas/latency_lut.csv', index=False)
