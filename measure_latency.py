import torch
import torch.nn as nn
import time
from blocks import ChannelMask, FBNetV2BasicSearchBlock

def measure_latency(operation, num_runs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    operation = operation.to(device)
    input_tensor = torch.randn(operation.input_shape).to(device)

    torch.cuda.synchronize()  # Synchronize for accurate timing
    start_time = time.time()

    for _ in range(num_runs):
        _ = operation(input_tensor)
        torch.cuda.synchronize()

    end_time = time.time()
    latency = (end_time - start_time) / num_runs

    return latency * 1000

if 0:
# Define operations and configurations
    operations = [
        {'type': 'conv', 'kernel_size': 3, 'stride': 1, 'in_channels': 16, 'out_channels': 16}, # warm up
        {'type': 'conv', 'kernel_size': 3, 'stride': 1, 'in_channels': 16, 'out_channels': 16},
        {'type': 'conv', 'kernel_size': 3, 'stride': 2, 'in_channels': 16, 'out_channels': 16},
        {'type': 'conv', 'kernel_size': 5, 'stride': 1, 'in_channels': 16, 'out_channels': 16},
        {'type': 'conv', 'kernel_size': 5, 'stride': 2, 'in_channels': 16, 'out_channels': 16},
    ]

    # Measure latencies
    latency_lut = {}
    for operation_config in operations:
        if operation_config['type'] == 'conv':
            conv = nn.Conv2d(operation_config['in_channels'], operation_config['out_channels'], operation_config['kernel_size'], operation_config['stride'])
            conv.input_shape = (1, operation_config['in_channels'], 32, 32)
            latency = measure_latency(conv)
            key = (operation_config['type'], operation_config['kernel_size'], operation_config['stride'], operation_config['in_channels'], operation_config['out_channels'])
            latency_lut[key] = latency

    print(latency_lut)

"""
Measure Channel Masking
"""
if 1:
    configs = [
        {'type': 'channel_mask', 'in_channels': 3, 'max_out_channels': 10, 'num_masks': 3}, # warm up
    ]
    for in_channels in range(1,300):
        for max_out_channels in range(1,300):
            for num_masks in range(1,20):
                configs.append({'type': 'channel_mask', 'in_channels': in_channels, 'max_out_channels': max_out_channels, 'num_masks': num_masks})

    input_configs = [
        {'type': 'tensor', 'batch_size': 1, 'depth': 1, 'height': 3, 'width': 3},
    ]
    for depth in range(1,300):
        for height in range(1,1024):
            for width in range(1,1024):
                input_configs.append({'type': 'tensor', 'batch_size': 1, 'depth': depth, 'height': height, 'width': width})

    latency_lut = {}
    for
    for operation_config in operations:
        if operation_config['type'] == 'channel_mask':
            op = ChannelMask(operation_config['in_channels'], operation_config['max_out_channels'], operation_config['num_masks'])
            conv.input_shape = (1, operation_config['in_channels'], 32, 32)
            latency = measure_latency(conv)
            key = (operation_config['type'], operation_config['kernel_size'], operation_config['stride'], operation_config['in_channels'], operation_config['out_channels'])
            latency_lut[key] = latency

    print(latency_lut)
