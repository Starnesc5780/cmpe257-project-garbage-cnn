import torch.nn as nn
from torchvision import models

def get_finetune_model(num_classes=9):
    # Load a pre-trained ResNet-18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Optional: Freeze the parameters so we only train the final layer
    # For better accuracy, we can leave them unfrozen to fine-tune the whole network,
    # but let's freeze early layers and only train the last block and fc for speed/stability
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the last convolutional block (layer4) for better fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model
