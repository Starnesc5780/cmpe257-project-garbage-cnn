import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

# Add src to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing.process_data import preprocess_data, load_dataset
from src.experiment4.transfer_model import get_finetune_model

def run_camera():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the Model 
    MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "experiment4", "models", "transfer_cnn_augmented.pth")
    
    base_dataset = load_dataset()
    classes = base_dataset.classes

    model = get_finetune_model(num_classes=len(classes)).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train Experiment 4 first!")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 2. Setup Preprocessing
    _, eval_transform = preprocess_data(use_augmentation=False)

    # 3. Open Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert OpenCV BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Preprocess and prepare for model
        input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

        # UI Overlay
        label = classes[predicted_idx]
        conf_score = confidence.item() * 100
        
        # Color changes based on confidence (Green if > 70%, else Yellow/Red)
        color = (0, 255, 0) if conf_score > 70 else (0, 255, 255)
        
        text = f"{label}: {conf_score:.1f}%"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Garbage Classification - Real Time', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()
