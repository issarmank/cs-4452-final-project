import torch
import torch.nn as nn
import torch.optim as optim

class DynamicNN(nn.Module):
    def __init__(self, size_per_layer) -> None:
        """
        Creating a dynamic neural network that will have a variable number of layers and neurons per layer.

        Args:
            size_per_layer (list): A list containing the number of neurons for each layer, including input and output layers.
        """
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(size_per_layer[i], size_per_layer[i + 1]) for i in range(len(size_per_layer) - 1)])
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
    
    def forward(self, x):
        x = self.flat(x)
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < len(self.linears) - 1:
                x = self.relu(x)
        return x