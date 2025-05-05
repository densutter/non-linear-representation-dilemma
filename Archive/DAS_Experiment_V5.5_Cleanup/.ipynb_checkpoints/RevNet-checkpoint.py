import torch.nn as nn
import torch

class RevNet_Block(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(RevNet_Block, self).__init__()
        self.half_in_features=in_features//2
        self.F_1 = nn.Linear(self.half_in_features, hidden_size)
        self.F_2 = nn.Linear(hidden_size, self.half_in_features)
        self.G_1 = nn.Linear(self.half_in_features, hidden_size)
        self.G_2 = nn.Linear(hidden_size, self.half_in_features)
        self.act = nn.ReLU()

    def forward(self, x):
        x_1 = x[:,:self.half_in_features]
        x_2 = x[:,self.half_in_features:]
        F_O = self.F_1(x_2)
        F_O = self.act(F_O)
        F_O = self.F_2(F_O)
        y_1 = x_1 + F_O
        G_O = self.G_1(y_1)
        G_O = self.act(G_O)
        G_O = self.G_2(G_O)
        y_2 = x_2 + G_O
        y   = torch.cat((y_1, y_2), dim=1)
        return y

    def inverse(self, y):
        y_1 = y[:,:self.half_in_features]
        y_2 = y[:,self.half_in_features:]
        G_O = self.G_1(y_1)
        G_O = self.act(G_O)
        G_O = self.G_2(G_O)
        x_2 = y_2 - G_O
        F_O = self.F_1(x_2)
        F_O = self.act(F_O)
        F_O = self.F_2(F_O)
        x_1 = y_1 - F_O
        x   = torch.cat((x_1, x_2), dim=1)
        return x



class RevNet(nn.Module):
    def __init__(self, number_blocks, in_features, hidden_size):
        super(RevNet, self).__init__()
        Model_Layers = []
        for i in range(number_blocks):
            Model_Layers.append(RevNet_Block(in_features, hidden_size))
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








