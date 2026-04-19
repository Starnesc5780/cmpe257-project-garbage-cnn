# Model: Base CNN model used in the project
# Data Preprocessing: Image resizing (if needed)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A custom Residual Block that adds the input (shortcut) to the output of the convolutional layers.
    This solves the vanishing gradient problem and allows training of much deeper networks.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        # If dimensions change, we need a 1x1 convolution to match dimensions before adding
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) 
        out = F.relu(out)
        return out

class BaseGarbageCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(BaseGarbageCNN, self).__init__()
        # Input size: (3, 224, 224)
        
        # Initial Convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual Blocks
        self.layer1 = ResidualBlock(32, 64, stride=2)   # Output: (64, 112, 112)
        self.layer2 = ResidualBlock(64, 128, stride=2)  # Output: (128, 56, 56)
        self.layer3 = ResidualBlock(128, 256, stride=2) # Output: (256, 28, 28)
        self.layer4 = ResidualBlock(256, 512, stride=2) # Output: (512, 14, 14)
        
        # Global Average Pooling (drastically reduces parameters to 512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classifier
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
