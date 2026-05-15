import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from data_processing.process_data import *
from transfer_model import get_finetune_model

train_transform, evaluation_transform = preprocess_data(use_augmentation=True)

base_dataset = load_dataset(transform=evaluation_transform)
train_indices, val_indices, test_indices = split_dataset_indices(base_dataset)
train_data = create_augmented_dataset(train_indices, train_transform, evaluation_transform)
val_data = create_base_dataset(evaluation_transform, val_indices)
test_data = create_base_dataset(evaluation_transform, test_indices)

train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

num_classes = len(base_dataset.classes)
model = get_finetune_model(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

max_epochs = 20
patience = 2
best_val_loss = float("inf")
patience_counter = 0
best_state_dict = None

print(f"Training for {max_epochs} epochs...")

for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")
            
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"--- Epoch {epoch+1} Summary: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss_avg = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1} - Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")

    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        patience_counter = 0
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# 4. Evaluation
model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        
print(classification_report(test_labels, test_preds, target_names=base_dataset.classes, zero_division=0))

models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'transfer_cnn_augmented.pth')
torch.save(model.state_dict(), model_path)
print(f"Saved to {model_path}")
