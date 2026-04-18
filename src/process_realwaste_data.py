'''
Process RealWaste dataset and save it in a format suitable for training the CNN model.
Raw Data Notes:
- Images are already sized to 524x524 pixels
- Labels: 
    -Cardboard
    -Food Organics
    -Glass
    -Metal
    -Miscellaneous Trash
    -Paper
    -Plastic
    -Textile Trash
    -Vegetation
'''

#Imports
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

#Constants
import os
path_to_data = os.path.join(os.path.dirname(__file__), "datasets", "realwaste-main", "RealWaste")
image_size = 224 #resizing images to 224x224 for CNN input (original 524x524)
batch_size = 32

#Data Splits
training_ratio = 0.7
validation_ratio = 0.15
testing_ratio = 0.15

#Random Seed (designated so that we can reproduce the same splits across runs)
random_seed = 42

#Preprocessing: Convert Data to PyTorch Transforms
'''
transforms.Compose will also help with Data Augmentation in the future
For now, this is just resizing and normalization for the base CNN model
'''
def preprocess_data(use_augmentation=False):
    if use_augmentation:
        training_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        training_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    #Preprocessed Data Transform for Validation and Testing
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return training_transform, evaluation_transform

#Load Dataset into PyTorch Dataset Format
def load_dataset(transform=None):
    dataset = datasets.ImageFolder(root=path_to_data, transform=transform)
    return dataset

#Split PyTorch Dataset into Training, Validation, and Testing Sets
def split_dataset(data):
    #Compute sizes for data splits
    training_size = int(training_ratio * len(data))
    validation_size = int(validation_ratio * len(data))
    testing_size = len(data) - training_size - validation_size

    #Randomly split dataset
    training_dataset, validation_dataset, testing_dataset = random_split(
        data,
        [training_size, validation_size, testing_size],
        generator=torch.Generator().manual_seed(random_seed),
    )
    return training_dataset, validation_dataset, testing_dataset

#Create PyTorch DataLoaders for Batching (and Shuffling for Training)
def create_dataloaders(training_dataset, validation_dataset, testing_dataset):
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
    return training_loader, validation_loader, testing_loader

