import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    """
    Batch iterator for a dataset represented as numpy arrays.

    Args:
        X: numpy array of shape (n_samples, n_features)
        y: Optional numpy array of shape (n_samples,)
        batch_size: Number of samples per batch

    Returns:
        List of Tensor batches (or [X_batch, y_batch] pairs if y is provided)
    """
    # 兼容输入：可以接收torch tensor或者numpy array
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X
    if y is not None:
        if isinstance(y, np.ndarray):
            y_tensor = torch.tensor(y, dtype=torch.float32)
        else:
            y_tensor = y
        dataset = TensorDataset(X_tensor, y_tensor)
    else:
        dataset = TensorDataset(X_tensor)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    batches = []
    for batch in loader:
        if y is not None:
            batches.append(batch) # (Xb, yb)
        else:
            batches.append(batch[0]) # 取出里面的tensor，丢掉外层元组
    return batches
