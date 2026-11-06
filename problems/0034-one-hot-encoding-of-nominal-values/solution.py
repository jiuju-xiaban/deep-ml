import torch
from typing import Optional

def to_categorical(x: torch.Tensor, n_col: Optional[int] = None) -> torch.Tensor:
    """
    Perform one-hot encoding on a 1D integer tensor `x`. If `n_col` is not provided, infer it from the max value in `x`.
    """
    # Hint: You can use torch.nn.functional.one_hot
    if n_col is None:
        n_col=int(torch.max(x).item())+1
    # return torch.nn.functional.one_hot(x,num_classes=n_col).float()
    row=x.shape[0]
    output=torch.zeros(row,n_col,dtype=torch.float)
    output[torch.arange(row),x]=1.0
    return output
