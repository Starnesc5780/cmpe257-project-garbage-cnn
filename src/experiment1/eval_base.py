import sys
import torch
from torchvision import datasets

DATASET_PATH = "./data/raw/realwaste-main/RealWaste"
CHECKPOINT_PATH = "./src/experiment1/models/base_cnn_optimized.pth"

sys.path.insert(0, "./src")
from data_processing.process_data import BATCH_SIZE, preprocess_data
from analysis.eval_helpers import *
from basic_cnn import BaseGarbageCNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_transform, evaluation_transform = preprocess_data(use_augmentation=False)

    full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=evaluation_transform)
    test_loader = build_test_loader(full_dataset, batch_size=BATCH_SIZE)

    model = BaseGarbageCNN(num_classes=len(full_dataset.classes)).to(device)
    load_checkpoint(model, CHECKPOINT_PATH, device)

    test_labels, test_predictions = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(test_labels, test_predictions, full_dataset.classes)
    print_metrics("Experiment 1", CHECKPOINT_PATH, metrics)

if __name__ == "__main__":
    main()
