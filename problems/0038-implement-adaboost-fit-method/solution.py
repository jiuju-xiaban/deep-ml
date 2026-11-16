import torch
import math
from typing import List, Dict

def adaboost_fit(X, y, n_clf) -> List[Dict]:
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    n_samples, n_features = X.shape
    w = torch.full((n_samples,), 1.0 / n_samples)
    clfs = []
    for _ in range(n_clf):
        clf = {}
        min_error = float('inf')
        # Find best decision stump
        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            thresholds = torch.unique(feature_values)
            for threshold in thresholds:
                for polarity in [1, -1]:
                    preds = torch.ones(n_samples)
                    preds[polarity * feature_values < polarity * threshold] = -1
                    error = torch.sum(w * (preds != y))
                    if error > 0.5:
                        error = 1 - error
                        polarity = -polarity
                    if error < min_error:
                        min_error = error.item()
                        clf['polarity'] = polarity
                        clf['threshold'] = threshold.item()
                        clf['feature_index'] = feature_i
        # Compute alpha
        clf['alpha'] = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
        # Update weights
        feature_values = X[:, clf['feature_index']]
        preds = torch.ones(n_samples)
        polarity = clf['polarity']
        preds[polarity * feature_values < polarity * clf['threshold']] = -1
        w = w * torch.exp(-clf['alpha'] * y * preds)
        w = w / torch.sum(w)
        clfs.append(clf.copy())
    return clfs
