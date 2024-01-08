import torch

def get_device():
    """
    Determines the best available device (CUDA, MPS, or CPU)
    and returns it for later use with PyTorch.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
