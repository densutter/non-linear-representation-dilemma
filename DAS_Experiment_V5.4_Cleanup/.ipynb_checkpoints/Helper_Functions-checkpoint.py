import random
import numpy as np
import torch

def set_seed(seed):
    # Set the seed for the random module
    random.seed(seed)
    
    # Set the seed for numpy
    np.random.seed(seed)
    
    # Set the seed for PyTorch (CPU)
    torch.manual_seed(seed)
    
    # Set the seed for PyTorch (GPU) if you are using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True  # This makes the computations deterministic
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning for performance optimization
    
    # For reproducibility of other libraries like Python's `random`
    torch.random.manual_seed(seed)




