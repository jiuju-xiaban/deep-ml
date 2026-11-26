import torch

def gradient_descent(X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, 
                     learning_rate: float, n_epochs: int, 
                     batch_size: int = 1, method: str = 'batch') -> torch.Tensor:
    """
    Implements three variants of gradient descent: Batch, Stochastic, and Mini-Batch.
    Uses Mean Squared Error (MSE) as the loss function.
    
    Args:
      X: Feature matrix of shape (m, n)
      y: Target values of shape (m,)
      weights: Initial weights of shape (n,)
      learning_rate: Step size for gradient descent
      n_epochs: Number of complete passes through the dataset
      batch_size: Size of batches for mini-batch gradient descent (default: 1)
      method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
      Optimized weights as a tensor
    """
    w = weights.clone()
    m = X.shape[0]

    for _ in range(n_epochs):
        if method == "batch":
            # 全部样本作为一个batch
            Xb = X
            yb = y
            k = m
            yhat = Xb @ w
            residual = yhat - yb
            g = (2.0 / k) * Xb.T @ residual
            w = w - learning_rate * g

        elif method == "stochastic":
            # 逐个样本循环，batch size=1
            for idx in range(m):
                Xb = X[idx:idx+1]
                yb = y[idx:idx+1]
                k = 1
                yhat = Xb @ w
                residual = yhat - yb
                g = (2.0 / k) * Xb.T @ residual
                w = w - learning_rate * g

        elif method == "mini_batch":
            # 按顺序切不重叠batch
            for start in range(0, m, batch_size):
                end = start + batch_size
                Xb = X[start:end]
                yb = y[start:end]
                k = Xb.shape[0]
                yhat = Xb @ w
                residual = yhat - yb
                g = (2.0 / k) * Xb.T @ residual
                w = w - learning_rate * g
    return w
