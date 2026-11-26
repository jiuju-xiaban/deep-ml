import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n_samples = X.shape[0]
    w = weights.copy()  # 不要修改原数组
    
    for i in range(n_iter):
        # 选样本：i % n_samples，按顺序循环，不shuffle
        idx = i % n_samples
        xi = X[idx]
        yi = y[idx]
        
        # 预测 y_hat = xi · w
        y_hat = np.dot(xi, w)
        
        # 单个样本MSE对权重的梯度：∇ = 2*(yhat - yi)*xi
        grad = 2 * (y_hat - yi) * xi
        
        # SGD 更新：w = w - lr * grad
        w = w - learning_rate * grad
    
    # 转成Python list返回
    return w.tolist()
