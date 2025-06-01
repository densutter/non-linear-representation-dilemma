import random
import numpy as np
import torch
import os

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


def ensure_directory_for_file(file_path):
    """
    Ensures that the directory for the given file path exists.
    Creates the directory if it doesn't exist.
    
    Args:
        file_path (str): Path to the file for which directories should be ensured.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


