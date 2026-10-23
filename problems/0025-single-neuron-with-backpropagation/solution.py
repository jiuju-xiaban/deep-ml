import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    # 复制初始参数，不用autograd，手动求导更新
    w = initial_weights.clone().detach()
    b = torch.tensor(initial_bias, dtype=torch.float32)
    mse_values = []
    n_samples = features.shape[0]

    for _ in range(epochs):
        # 前向传播
        z = features @ w + b
        a = torch.sigmoid(z)
        loss = torch.mean((a - labels) ** 2)
        mse_values.append(round(loss.item(), 4))

        # 手动反向传播公式：MSE + sigmoid
        dL_dz = 2 * (a - labels) * a * (1 - a) / n_samples
        dL_dw = features.T @ dL_dz
        dL_db = torch.sum(dL_dz)

        # SGD 更新
        w = w - learning_rate * dL_dw
        b = b - learning_rate * dL_db

    # 转成要求格式，保留4位小数
    updated_weights = [round(x.item(), 4) for x in w]
    updated_bias = round(b.item(), 4)

    return updated_weights, updated_bias, mse_values
