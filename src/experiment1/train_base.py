import torch
import torch.nn as nn
import torch.optim as optim
from data_processing.process_data import *
from basic_cnn import BaseGarbageCNN
from sklearn.metrics import classification_report
import os

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
epochs = 18    # optimal number before overfitting
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
for epoch in range(epochs):
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
    
    if epoch == epochs - 1:
        print(classification_report(all_labels, all_preds, target_names=full_dataset.classes, zero_division=0))

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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'base_cnn_optimized.pth')
torch.save(model.state_dict(), model_path)

