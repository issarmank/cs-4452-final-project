import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from model import DynamicNN
from train_test_loop import train_model, test_model


# Hyperparameter search spaces
LR_SPACE = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
NUM_HIDDEN_LAYERS_SPACE = [1, 2, 3, 4]
HIDDEN_DIM_SPACE = [16, 32, 64, 128, 256]
BATCH_SIZE_SPACE = [16, 32, 64, 128, 256]

# Loading the MNIST dataset
train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

img, label = train_data[0]
INPUT_SIZE = img.shape[0] * img.shape[1] * img.shape[2]   # 784
OUTPUT_SIZE = 10


class TestHpo:

    def __init__(self):
        self.current_lr_index = 0
        self.current_num_hl_index = 0
        self.current_hl_dim_index = 0
        self.current_batch_size_index = 0

    def generate_hidden_layer_configs(self, num_hidden_layers, hidden_dim):
        """
        Build one architecture using:
        - input size
        - selected number of hidden layers
        - selected hidden dimension
        - output size
        """
        configs = [INPUT_SIZE]

        for _ in range(num_hidden_layers):
            configs.append(hidden_dim)

        configs.append(OUTPUT_SIZE)
        return configs

    def get_model_performance(self):
        batch_size = BATCH_SIZE_SPACE[self.current_batch_size_index]

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        hidden_layers = self.generate_hidden_layer_configs(
            NUM_HIDDEN_LAYERS_SPACE[self.current_num_hl_index],
            HIDDEN_DIM_SPACE[self.current_hl_dim_index]
        )

        model = DynamicNN(hidden_layers)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR_SPACE[self.current_lr_index])

        model = train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, 5)
        test_loss, test_acc = test_model(model, test_dataloader, loss_fn)