'''
This file is used to check if all the required libraries are installed and working 
properly. It imports the necessary modules and prints out some basic information 
to confirm that everything is set up correctly.

Current libraries checked:
- PyTorch (torch)
    - Neural network module (torch.nn)
    - Optimizers (torch.optim)
    - DataLoader (torch.utils.data.DataLoader)
- Torchvision 
    - Datasets (torchvision.datasets)
    - Transforms (torchvision.transforms)
    - Pretrained models (torchvision.models)
- NumPy (numpy)
- Matplotlib (matplotlib.pyplot)
'''

# PyTorch core
import torch
print(torch.__version__)

# Neural network module
import torch.nn as nn
print(nn.Linear(10, 2))

# Optimizers
import torch.optim as optim
print(optim.SGD([torch.randn(2, requires_grad=True)], lr=0.01))

# Torchvision datasets
from torchvision import datasets
print(datasets.ImageFolder)

# Torchvision transforms (data augmentation later)
from torchvision import transforms
print(transforms.Resize((224, 224)))

# DataLoader
from torch.utils.data import DataLoader
print(DataLoader)

# Pretrained models (for transfer learning later)
from torchvision import models
print(models.resnet18)

# NumPy (optional but useful)
import numpy as np
print(np.array([1, 2, 3]))

# Visualization
import matplotlib.pyplot as plt
fig = plt.figure()
print(fig)
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Matplotlib test plot")
plt.show(block=True)