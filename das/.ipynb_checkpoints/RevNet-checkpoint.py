import torch.nn as nn
import torch


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) with configurable depth and width.
    
    This implementation allows for dynamic specification of the network architecture
    through the hidden_sizes parameter, which determines both the width and depth
    of the network.
    
    Args:
        input_size (int): Dimensionality of the input features
        output_size (int): Dimensionality of the output features
        hidden_sizes (list of int): List specifying the size of each hidden layer
        activation (nn.Module, optional): Activation function to use between layers.
                                         Defaults to nn.ReLU()
        dropout_rate (float, optional): Dropout probability. Defaults to 0.0
    """
    def __init__(self, input_size, output_size, hidden_sizes, activation=nn.ReLU(), dropout_rate=0.0):
        super(MLP, self).__init__()
        
        # Validate inputs
        if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must be a non-empty list of integers")
        
        # Build the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the MLP."""
        return self.network(x)


class RevNet_Block(nn.Module):
    def __init__(self, in_features, hidden_size, depth=1):
        super(RevNet_Block, self).__init__()
        self.half_in_features=in_features//2

        self.F = MLP(self.half_in_features, self.half_in_features, [hidden_size]*depth)
        self.G = MLP(self.half_in_features, self.half_in_features, [hidden_size]*depth)

    def forward(self, x):
        x_1 = x[:,:self.half_in_features]
        x_2 = x[:,self.half_in_features:]
        F_O = self.F(x_2)
        y_1 = x_1 + F_O
        G_O = self.G(y_1)
        y_2 = x_2 + G_O
        y   = torch.cat((y_1, y_2), dim=1)
        return y

    def inverse(self, y):
        y_1 = y[:,:self.half_in_features]
        y_2 = y[:,self.half_in_features:]
        G_O = self.G(y_1)
        x_2 = y_2 - G_O
        F_O = self.F(x_2)
        x_1 = y_1 - F_O
        x   = torch.cat((x_1, x_2), dim=1)
        return x


class RevNet(nn.Module):
    def __init__(self, number_blocks, in_features, hidden_size, depth=1):
        super(RevNet, self).__init__()
        Model_Layers = []
        for i in range(number_blocks):
            Model_Layers.append(RevNet_Block(in_features, hidden_size, depth))
        self.Model_Layers = nn.ModuleList(Model_Layers)

    def forward(self, x):
        for ac_layer in self.Model_Layers:
            x = ac_layer(x)
        return x

    def inverse(self, y):
        """Applies inverse transformation with high precision."""
        for ac_layer in reversed(self.Model_Layers):
            y = ac_layer.inverse(y)
        return y








