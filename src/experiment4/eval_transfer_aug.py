import sys
import os
import torch
from torchvision import datasets

# Set up path to include src root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Path to the data (relative to project root)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "realwaste-main", "RealWaste")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "transfer_cnn_augmented.pth")

from src.data_processing.process_data import BATCH_SIZE, preprocess_data
from src.analysis.eval_helpers import *
from src.experiment4.transfer_model import get_finetune_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, evaluation_transform = preprocess_data(use_augmentation=False)

full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=evaluation_transform)
test_loader = build_test_loader(full_dataset, batch_size=BATCH_SIZE)

model = get_finetune_model(num_classes=len(full_dataset.classes)).to(device)

if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    sys.exit(1)

load_checkpoint(model, MODEL_PATH, device)
test_labels, test_predictions = evaluate_model(model, test_loader, device)
metrics = compute_metrics(test_labels, test_predictions, full_dataset.classes)

print_metrics("Exp 4 (Transfer+Aug)", MODEL_PATH, metrics)
