import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

# Constants
DATA_PATH = "../../data/raw/realwaste-main/RealWaste"
IMAGE_SIZE = 224
BATCH_SIZE = 32 #standard batch size for training CNNs
TRAINING_RATIO = 0.7
VALIDATION_RATIO = 0.15
TESTING_RATIO = 0.15
RANDOM_SEED = 42


def preprocess_data(use_augmentation=False):
	if use_augmentation == True:
		training_transform = transforms.Compose(
			[
				transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)), 
				transforms.RandomHorizontalFlip(),
				transforms.RandomRotation(15),
				transforms.ToTensor(),
				transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
			]
		)
	else:
		training_transform = transforms.Compose(
			[transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
		)

	evaluation_transform = transforms.Compose(
		[transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
	)
	return training_transform, evaluation_transform


def load_dataset(transform=None):
	return datasets.ImageFolder(root=DATA_PATH, transform=transform)


#Stores the indices to keep splits consistent across each experiment
def split_dataset_indices(data):
	train_size = int(TRAINING_RATIO * len(data))
	val_size = int(VALIDATION_RATIO * len(data))
	test_size = len(data) - train_size - val_size

    #Create list of shuffled indices
	generator = torch.Generator().manual_seed(RANDOM_SEED)
	shuffled_indices = torch.randperm(len(data), generator=generator).tolist()

	training_indices = shuffled_indices[:train_size]
	validation_indices = shuffled_indices[train_size:(train_size+val_size)]
	testing_indices = shuffled_indices[(train_size+val_size):]
	return training_indices, validation_indices, testing_indices


#Only used for Data Augmentation Experiment's training set
def create_augmented_dataset(training_indices, training_transform, evaluation_transform):
	base_dataset = load_dataset(transform=evaluation_transform)
	augmented_dataset = load_dataset(transform=training_transform)
	base_subset = Subset(base_dataset, training_indices)
	augmented_subset = Subset(augmented_dataset, training_indices)
	return ConcatDataset([base_subset, augmented_subset])


#Used for training sets in Exp 1 and 2, and for validation set in all experiments
def create_base_dataset(transform, indices):
	dataset = load_dataset(transform=transform)
	return Subset(dataset, indices)


def create_dataloaders(training_dataset, validation_dataset, testing_dataset):
	training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
	validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
	testing_loader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=False)
	return training_loader, validation_loader, testing_loader
