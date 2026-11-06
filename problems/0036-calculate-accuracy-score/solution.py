import torch
from typing import Union

def accuracy_score(y_true: Union[torch.Tensor, list, "np.ndarray"],
                   y_pred: Union[torch.Tensor, list, "np.ndarray"]) -> float:
    """
    Compute the accuracy: fraction of matching elements in y_true and y_pred.
    Both inputs may be torch.Tensor, list, or numpy.ndarray.
    """
    # Your implementation here
    y_pred=torch.as_tensor(y_pred)
    y_true=torch.as_tensor(y_true)
    right=y_pred==y_true
    return torch.sum(right).item()/y_pred.shape[0]
