import torch
import torch.nn as nn
import torch.optim as optim
from process_realwaste_data import preprocess_data, load_dataset, split_dataset, create_dataloaders
from transfer_model import get_finetune_model
from sklearn.metrics import classification_report
import os

def main():
    # 1. Preprocessing and Data Loading
    print("Setting up data preprocessing...")
    train_transform, eval_transform = preprocess_data()
    
    # Since transforms are currently identical, we apply the training_transform to the whole dataset.
    print("Loading dataset...")
    try:
        full_dataset = load_dataset(transform=train_transform)
        print(f"Dataset loaded with {len(full_dataset)} images.")
        print(f"Classes: {full_dataset.classes}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("Splitting dataset...")
    train_data, val_data, test_data = split_dataset(full_dataset)
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)

    # 2. Model, Loss, Optimizer setup
    print("Initializing Base CNN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = len(full_dataset.classes)
    model = get_finetune_model(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop (Baseline)
    epochs = 3
    print(f"Starting training for {epochs} epochs...")
    
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
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")
                
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
        print("\nClassification Report (Precision, Recall, F1-Score):")
        print(classification_report(all_labels, all_preds, target_names=full_dataset.classes, zero_division=0))

    print("Fine-tuning complete!")
    
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
    print(classification_report(test_labels, test_preds, target_names=full_dataset.classes, zero_division=0))
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'transfer_cnn.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    main()
