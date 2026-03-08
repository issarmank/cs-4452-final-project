import torch
import torch.nn as nn

import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from model import DynamicNN
from train_test_loop import train_model, test_model

#Loading the MNIST dataset
train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

BATCH_SIZE = 64
img, label = train_data[0]
INPUT_SIZE = img.shape[0] * img.shape[1] * img.shape[2]  # Flattened size for input to basic neural network, 784 since 1x28x28  
NUM_CLASSES = len(train_data.classes) # 10 output classes for MNIST

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

hidden_layers = [784, 64, 32, 16, 10]
model = DynamicNN(hidden_layers)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model = train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, 5)
test_model(model, test_dataloader, loss_fn)

