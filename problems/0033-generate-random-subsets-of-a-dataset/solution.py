import numpy as np
from typing import List, Tuple

def get_random_subsets(X, y, n_subsets, replacements=True) -> list:
    """
    Generate n_subsets random subsets from the dataset (X, y).
    Each subset is a tuple (X_subset, y_subset), where both are lists.
    
    Args:
        X: 2D array of shape (n_samples, n_features)
        y: 1D array of shape (n_samples,)
        n_subsets: Number of subsets to generate
        replacements: If True, sample with replacement
                      If False, sample without replacement
    
    Returns:
        List of (X_subset, y_subset) tuples
    """
    # Your code here
    n=X.shape[0]
    if replacements:
        subsets_size=n
    else:
        subsets_size=n//2
    subsets=[]
    for _ in range(n_subsets):
        indices=np.random.choice(n, subsets_size,replace=replacements)
        X_sub=X[indices].tolist()
        y_sub=y[indices].tolist()
        subsets.append((X_sub,y_sub))
    return subsets
