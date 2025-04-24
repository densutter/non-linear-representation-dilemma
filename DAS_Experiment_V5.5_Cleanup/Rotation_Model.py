import torch.nn as nn
import torch


class RotateLayer(nn.Module):
    """A learnable linear transformation initialized as an orthogonal matrix."""

    def __init__(self, n, init_orth=True):
        """
        Args:
            n (int): Dimension of the square transformation matrix.
            init_orth (bool): If True, initializes the matrix with an orthogonal weight.
        """
        super().__init__()
        weight = torch.empty(n, n)  # Create an empty n x n matrix
        
        # We don't need initialization if loading from a pretrained checkpoint.
        # You can explore different initialization strategies if necessary, but this isn't our focus.
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        
        self.weight = torch.nn.Parameter(weight, requires_grad=True)  # Learnable weight matrix

    def forward(self, x):
        """Applies the rotation matrix to the input tensor."""
        return torch.matmul(x.to(self.weight.dtype), self.weight)
        

class Rotation(nn.Module):
    """Encapsulates the rotation transformation as a PyTorch module."""

    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): The embedding dimension (size of the transformation matrix).
        """
        super(Rotation, self).__init__()
        
        rotate_layer = RotateLayer(embed_dim)  # Initialize the rotation layer
        # Ensure the transformation remains an orthogonal matrix
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)

    def forward(self, x):
        """Applies the orthogonal transformation to the input tensor."""
        return self.rotate_layer(x)
    
    def inverse(self, x):
        weight_T = self.rotate_layer.weight.T  # Use matrix transpose as inverse
        return torch.matmul(x.to(weight_T.dtype), weight_T)




