'''
Process RealWaste dataset and save it in a format compatible with PyTorch
Raw Data Notes:
-Images are already sized to 524x524 pixels
-Labels: (Manual renaming of some labels for simplicity)
    -Cardboard
    -Food Organics -> Organics
    -Glass
    -Metal
    -Miscellaneous Trash -> Miscellaneous
    -Paper
    -Plastic
    -Textile Trash -> Clothing
    -Vegetation
-Counts: Total 4761 images, with an average of 529 examples per label (9 labels total)
'''

#Imports
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

#Constants
path_to_data = "../../data/raw/realwaste-main/RealWaste"
image_size = 224 #resizing images to image_size by image_size for CNN input (original 524x524)
batch_size = 32 #since our dataset is relatively small in size, we should use batch size 32 or 64

#Data Splits
training_ratio = 0.7
validation_ratio = 0.15
testing_ratio = 0.15

#Random Seed (designated so that we can reproduce the same splits across runs)
random_seed = 42

#Preprocessing: Convert Data to PyTorch Transforms w/ Data Augmentation
'''
-transforms.Compose config will also help with Data Augmentation in the future
    -Image flipping, rotating, color jitter
-For now, this is just resizing and normalization for the base CNN model
'''
# def augment_data():
#     #Preprocessed Data Transform
#     augmented_transform = transforms.Compose(
#         [
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ]
#     )
#     #Future implementation will have a separate train_transform with data augmentation
#     return augmented_transform

#Load Dataset into PyTorch Dataset Format
def load_dataset():
    dataset = datasets.ImageFolder(root=path_to_data)
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

