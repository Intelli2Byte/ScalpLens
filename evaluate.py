import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

MODEL_PATH = "./saved_models/best_scalp_model.pth"
VAL_DIR = "./dataset/val"
OUTPUT_CM_PATH = "./confusion_matrix.png"

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model checkpoint not found at {MODEL_PATH}")
        print("Please run 'python train.py' first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation Device: {device}")
    print(f"[*] Loading model checkpoint from: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint['classes']
    saved_acc = checkpoint.get('val_acc', 0.0)
    print(f"[*] Classes in Checkpoint: {classes}")
    print(f"[*] Checkpoint Saved Validation Accuracy: {saved_acc:.2f}%")

    # Load EfficientNet-B0 architecture
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, len(classes))
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Validation transforms
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    num_workers = 0 if os.name == 'nt' and not torch.cuda.is_available() else 2
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)

    y_true = []
    y_pred = []
    y_probs = []

    print("[*] Running inference on validation dataset...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    val_acc = (np.sum(y_true == y_pred) / len(y_true)) * 100

    print("\n" + "="*60)
    print("            CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print(f"Overall Validation Accuracy: {val_acc:.2f}%\n")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6), dpi=300)
    display_labels = [c.replace('_', ' ').title() for c in classes]
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(f"Scalp Pathology Confusion Matrix\nValidation Accuracy: {val_acc:.2f}%", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_CM_PATH, dpi=300)
    plt.close()

    print(f"[OK] Confusion matrix saved successfully to: {OUTPUT_CM_PATH}")

if __name__ == "__main__":
    evaluate_model()
