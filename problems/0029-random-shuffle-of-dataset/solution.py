import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
	if seed is not None:
        np.random.seed(seed)
    n = X.shape[0]
    perm_idx = np.random.permutation(n)
    return X[perm_idx], y[perm_idx]