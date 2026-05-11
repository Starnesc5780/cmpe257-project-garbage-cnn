import torch
import torch.nn as nn
import torch.optim as optim
from data_processing.process_data import *
from aug_cnn import BaseGarbageCNN
from sklearn.metrics import classification_report
import os

def main():
    print("Setting up data preprocessing (With Augmentation)...")
    train_transform, evaluation_transform = preprocess_data(use_augmentation=True)

    base_dataset = load_dataset(transform=evaluation_transform)
    train_indices, val_indices, test_indices = split_dataset_indices(base_dataset)

    train_data = create_augmented_dataset(train_indices, train_transform, evaluation_transform)
    val_data = create_base_dataset(evaluation_transform, val_indices)
    test_data = create_base_dataset(evaluation_transform, test_indices)

    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = len(base_dataset.classes)
    model = BaseGarbageCNN(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3
    print(f"Starting Data-Augmented Base CNN training for {epochs} epochs...")
    
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
        
        print(f"--- Epoch {epoch+1} Validation: Loss: {val_loss/len(val_loader):.4f}, Acc: {100 * val_correct / val_total:.2f}%")
        if epoch == epochs - 1:
            print("\nClassification Report (Augmented Base CNN):")
            print(classification_report(all_labels, all_preds, target_names=base_dataset.classes, zero_division=0))

    print("Augmented Base CNN training complete!")
    
    # Evaluate on Test Data
    print("\nEvaluating on Test Data...")
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
    print(f"FINAL TEST ACCURACY: {test_acc:.2f}%\n")
    print("Test Data Classification Report (Precision, Recall, F1-Score):")
    print(classification_report(test_labels, test_preds, target_names=base_dataset.classes, zero_division=0))
    
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'base_cnn_augmented.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    main()
