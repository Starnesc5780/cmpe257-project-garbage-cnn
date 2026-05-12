import sys
import torch
from torchvision import datasets

DATA_PATH = "./data/raw/realwaste-main/RealWaste"
MODEL_PATH = "./src/experiment2/models/transfer_cnn.pth"

sys.path.insert(0, "./src")
from data_processing.process_data import BATCH_SIZE, preprocess_data
from analysis.eval_helpers import *
from transfer_model import get_finetune_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_transform, evaluation_transform = preprocess_data(use_augmentation=False)

    full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=evaluation_transform)
    test_loader = build_test_loader(full_dataset, batch_size=BATCH_SIZE)

    model = get_finetune_model(num_classes=len(full_dataset.classes)).to(device)
    load_checkpoint(model, MODEL_PATH, device)

    test_labels, test_predictions = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(test_labels, test_predictions, full_dataset.classes)
    print_metrics("Experiment 2", MODEL_PATH, metrics)


if __name__ == "__main__":
    main()
