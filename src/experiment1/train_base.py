import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing.process_data import *
from src.experiment1.basic_cnn import BaseGarbageCNN

#export prints to file
sys.stdout = open('models/base_cnn_training_log.txt', 'w')

train_transform, evaluation_transform = preprocess_data(use_augmentation=False)
full_dataset = load_dataset(transform=evaluation_transform)
train_idx, val_idx, test_idx = split_dataset_indices(full_dataset)
train_data = create_base_dataset(evaluation_transform, train_idx)
val_data = create_base_dataset(evaluation_transform, val_idx)
test_data = create_base_dataset(evaluation_transform, test_idx)
train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #needed for gpu training
print(f"gpu check: {device}")
num_classes = len(full_dataset.classes)
model = BaseGarbageCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
max_epochs = 20
patience = 8
best_val_loss = float("inf")
patience_counter = 0
best_state_dict = None
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
start_time = time.time()
best_epoch_time = start_time
best_epoch = 0
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
            
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"epoch {epoch+1} train, loss: {epoch_loss:.4f}, acc: {epoch_acc:.2f}%")
    scheduler.step()
    model.eval() #evaluation for epoch step
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
    
    val_acc = 100 * val_correct / val_total
    val_loss_avg = val_loss/len(val_loader)
    print(f"epoch {epoch+1} val, loss: {val_loss_avg:.4f}, acc: {val_acc:.2f}%")

    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        patience_counter = 0
        best_epoch = epoch + 1
        best_epoch_time = time.time()
        best_state_dict = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"early stopping: best epoch found at epoch {best_epoch} (best val loss: {best_val_loss:.4f})")
            break

    if epoch == max_epochs - 1:
        print(classification_report(all_labels, all_preds, target_names=full_dataset.classes, zero_division=0))

end_time = time.time()
print(f"Model 1 training completed in {(end_time - start_time) / 60:.2f} minutes")
print(f"Model 1 best epoch found after {(best_epoch_time - start_time) / 60:.2f} minutes")

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

print("\nTest Data Step")
model.eval()
test_correct = 0
test_total = 0
test_preds = []
test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)   
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item() 
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        
test_acc = 100 * test_correct / test_total
print(f"final test accuracy: {test_acc:.2f}%\n")
print(classification_report(test_labels, test_preds, target_names=full_dataset.classes, zero_division=0))
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'base_cnn_optimized.pth')
torch.save(model.state_dict(), model_path)

