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
path_to_data = "/data/raw/realwaste-main"
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
transforms.Compose will also help with Data Augmentation
For now, this is just resizing and normalization for the base CNN model
'''
def preprocess_data():
    #Preprocessed Data Transform for Training
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