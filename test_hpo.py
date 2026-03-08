import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from model import DynamicNN
from train_test_loop import train_model, test_model
import time


# Hyperparameter search spaces
LR_SPACE = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
NUM_HIDDEN_LAYERS_SPACE = [1, 2, 3, 4]
HIDDEN_DIM_SPACE = [16, 32, 64, 128, 256]
BATCH_SIZE_SPACE = [32, 64, 128, 256]

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

        # keep epochs to 3 so that model training does not take too long...
        model = train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, 3)
        test_loss, test_acc = test_model(model, test_dataloader, loss_fn)

        return test_acc


    def test_grid_search_performance(self): 
        start_time = time.perf_counter()

        # cache for best performance
        best_test_acc = 0
        configuration = None

        # configuration 
        # since this is an exhaustive search, here we define 4 for loop 
        # which iterate through each combination of our hyperparameter space

        for lr_index in range(len(LR_SPACE)):
            for hl_num_space_index in range(len(NUM_HIDDEN_LAYERS_SPACE)):
                for hl_dim_space_index in range(len(HIDDEN_DIM_SPACE)):
                    for batch_size_space_index in range(len(BATCH_SIZE_SPACE)):
                        self.current_lr_index = lr_index
                        self.current_num_hl_index = hl_num_space_index
                        self.current_hl_dim_index = hl_dim_space_index
                        self.current_batch_size_index = batch_size_space_index

                        test_acc= self.get_model_performance()

                        if test_acc > best_test_acc:
                            
                            best_test_acc = test_acc
                            configuration = [
                                LR_SPACE[lr_index],
                                NUM_HIDDEN_LAYERS_SPACE[hl_num_space_index],
                                HIDDEN_DIM_SPACE[hl_dim_space_index],
                                BATCH_SIZE_SPACE[batch_size_space_index]
                            ]
        
        # Record the end time
        end_time = time.perf_counter()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print("HPO using Grid Search complete...")
        print(f"This algorithm took {elapsed_time:.4f} seconds")
        print(f"The best test accuracy found was {best_test_acc}")
        print(f"The configuration that performed the best was : ")
        print(f"lr = {configuration[0]}")
        print(f"Number of hidden layers = {configuration[1]}")
        print(f"Dimension of hidden layers = {configuration[2]}")
        print(f"Batch Size = {configuration[3]}")

        return elapsed_time, best_test_acc, configuration