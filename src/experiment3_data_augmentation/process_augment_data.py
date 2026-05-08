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
from torch.utils.data import DataLoader, ConcatDataset, Subset
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
def preprocess_data(use_augmentation=True):
    if use_augmentation:
        training_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)), #random zoom in/crop
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    # Preprocessed Data Transform for Validation and Testing
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

    #Randomly split indices so we can reuse the same examples with different transforms
    generator = torch.Generator().manual_seed(random_seed)
    shuffled_indices = torch.randperm(len(data), generator=generator).tolist()

    training_indices = shuffled_indices[:training_size]
    validation_indices = shuffled_indices[training_size : training_size + validation_size]
    testing_indices = shuffled_indices[training_size + validation_size :]

    return training_indices, validation_indices, testing_indices


#Concatenates the original data with the transformed data to create an augmented dataset
def create_training_dataset(training_indices, training_transform, evaluation_transform):
    original_dataset = load_dataset(transform=evaluation_transform)
    augmented_dataset = load_dataset(transform=training_transform)
    original_training_subset = Subset(original_dataset, training_indices)
    augmented_training_subset = Subset(augmented_dataset, training_indices)
    return ConcatDataset([original_training_subset, augmented_training_subset])


#Evaluation dataset only uses original dataset (no transformations)
def create_evaluation_dataset(transform, indices):
    dataset = load_dataset(transform=transform)
    return Subset(dataset, indices)

#Create PyTorch DataLoaders for Batching (and Shuffling for Training)
def create_dataloaders(training_dataset, validation_dataset, testing_dataset):
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
    return training_loader, validation_loader, testing_loader

