import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    scores=torch.as_tensor(scores,dtype=torch.float32)
    temp=torch.exp(scores)
    output=temp/torch.sum(temp)
    return output.tolist()
