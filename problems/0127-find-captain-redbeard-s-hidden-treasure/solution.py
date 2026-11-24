import torch

def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

    Args:
        start_x: Starting x position for optimization

    Returns:
        float: The x-coordinate of the minimum point.
    """
    # Your code here
    x=torch.tensor(start_x,requires_grad=True)
    lr=0.01
    iteritions=10000
    momentum=0.9
    v=torch.tensor(0.0)
    for _ in range(iteritions):
        f=x**4-3*x**3+2
        f.backward()
        with torch.no_grad():
            v=momentum*v-lr*x.grad
            x+=v
            x.grad.zero_()
    
    return float(x.item())
