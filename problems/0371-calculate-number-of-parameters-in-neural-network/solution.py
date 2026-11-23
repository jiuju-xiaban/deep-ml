import torch
import torch.nn as nn

def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network
    using PyTorch's built-in nn modules.

    Args:
        layers: List of dictionaries, each describing a layer.
                Supported types: 'dense' (uses nn.Linear), 'conv2d' (uses nn.Conv2d).

    Returns:
        Total number of trainable parameters (int).
    """
    # Your code here
    total_params=0
    for layer in layers:
        has_bias=layer.get('bias',True)
        if layer["type"]=='dense':
            module=nn.Linear(
                in_features=layer['input_size'],
                out_features=layer['output_size'],
                bias=has_bias
            )
        elif layer['type']=='conv2d':
            module=nn.Conv2d(
                in_channels=layer['in_channels'],
                out_channels=layer['out_channels'],
                kernel_size=layer['kernel_size'],
                bias=has_bias
            )
        else:
            continue
        total_params+=sum(p.numel() for p in module.parameters())
    return total_params
