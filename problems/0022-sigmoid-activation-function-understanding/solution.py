import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    z=torch.as_tensor(z)
    output=1/(1+torch.exp(-z))
    return float(torch.round(output*10000)/10000)
