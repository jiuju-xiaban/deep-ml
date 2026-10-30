import torch
from typing import List

def divide_on_feature(X, feature_i, threshold) -> List[torch.Tensor]:
    """
    Divide the tensor X into two subsets based on whether X[:, feature_i] is greater than or equal to (or equal to) the threshold.
    Return two tensors: one with samples that meet the condition, one with samples that do not.
    """
    X=torch.as_tensor(X)
    feature_i=torch.as_tensor(feature_i)
    threshold=torch.as_tensor(threshold)
    mask=X[:,feature_i]>=threshold
    first=torch.where(mask)[0][0]
    return X[first:,:],X[:first,:]
