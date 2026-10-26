import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"[*] Training on Device: {device}")

    # 1. Memory-Safe Preprocessing & Targeted Dermoscopic Augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder("./dataset/train", transform=train_transforms)
    val_dataset = datasets.ImageFolder("./dataset/val", transform=val_transforms)

    num_workers = 0 if os.name == 'nt' and not torch.cuda.is_available() else 2
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    print(f"[*] Mapped Classes: {class_names}")

    # 2. Build EfficientNet-B0 Architecture with Pretrained Weights
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, len(class_names))
    )
    model = model.to(device)

    # Class distribution weighting to handle class imbalance
    # Alopecia: 1141, Seborrheic: 602, Androgenetic: 496
    class_counts = [1141.0, 496.0, 602.0]  # sorted alphabetically matching class_names
    total_samples = sum(class_counts)
    weights_tensor = torch.tensor([total_samples / (len(class_counts) * c) for c in class_counts], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    os.makedirs("./saved_models", exist_ok=True)
    BEST_MODEL_PATH = "./saved_models/best_scalp_model.pth"

    EPOCHS = 10
    best_val_acc = 0.0

    # Progressive Staged Transfer Learning for Optimal Convergence & High Speed on CPU/GPU
    # Phase 1 (Epochs 1-3): Train classifier head with frozen backbone to avoid destructive gradient shock
    # Phase 2 (Epochs 4-7): Unfreeze top high-level semantic blocks (features[5:]) for domain adaptation
    # Phase 3 (Epochs 8-10): Fine-tune all layers with gentle cosine annealing
    print("\n[*] Initializing Staged Transfer Learning Pipeline...")

    for epoch in range(EPOCHS):
        if epoch == 0:
            print("  --> Phase 1: Training Classifier Head (Backbone Frozen)...")
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=8e-4, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

        elif epoch == 3:
            print("  --> Phase 2: Unfreezing High-Level Semantic Blocks (features[5:])...")
            for param in model.features[5:].parameters():
                param.requires_grad = True
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=1.5e-4, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

        elif epoch == 7:
            print("  --> Phase 3: Global Fine-Tuning across all layers...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

        # Training Epoch
        model.train()
        running_loss, train_correct = 0.0, 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_acc = (train_correct / total_train) * 100
        train_loss = running_loss / total_train

        # Validation Epoch
        model.eval()
        val_correct, total_val = 0, 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = (val_correct / total_val) * 100
        val_loss_avg = val_loss / total_val

        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': class_names,
                'val_acc': val_acc
            }, BEST_MODEL_PATH)
            print(f"  [+] Saved new checkpoint to {BEST_MODEL_PATH} (Accuracy: {val_acc:.2f}%)")

    print(f"\n[DONE] Training complete. Permanent weights saved at: {BEST_MODEL_PATH}")
    print(f"Peak Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()
