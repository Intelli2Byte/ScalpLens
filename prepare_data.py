import os
import sys
import shutil
import random
from PIL import Image

# Ensure standard UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Fixed seed for reproducibility
random.seed(42)

SOURCE_ROOT = "./archive/PUMCH-SCALP"
OUTPUT_DIR = "./dataset"

TARGET_CLASSES = [
    "Alopecia areata",
    "Seborrheic dermatitis",
    "Androgenetic alopecia"
]

def prepare_splits(train_ratio=0.8):
    for split in ["train", "val"]:
        for cls in TARGET_CLASSES:
            clean_cls = cls.lower().replace(" ", "_")
            os.makedirs(os.path.join(OUTPUT_DIR, split, clean_cls), exist_ok=True)

    print("[*] Processing and validating images...")
    total_train = 0
    total_val = 0

    for cls in TARGET_CLASSES:
        clean_cls = cls.lower().replace(" ", "_")
        src_folder = os.path.join(SOURCE_ROOT, cls)

        if not os.path.exists(src_folder):
            print(f"[!] Warning: Folder '{src_folder}' not found. Please verify directory name.")
            continue

        valid_images = []
        for file in os.listdir(src_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(src_folder, file)
                # Verify image integrity
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                    valid_images.append(file)
                except Exception:
                    continue

        random.shuffle(valid_images)
        split_idx = int(len(valid_images) * train_ratio)
        train_files = valid_images[:split_idx]
        val_files = valid_images[split_idx:]

        for f in train_files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(OUTPUT_DIR, "train", clean_cls, f))
        for f in val_files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(OUTPUT_DIR, "val", clean_cls, f))

        total_train += len(train_files)
        total_val += len(val_files)
        print(f"[OK] {cls} ({clean_cls}) -> Train: {len(train_files)} | Val: {len(val_files)} | Total: {len(valid_images)}")

    print(f"\n[SUMMARY] Data preparation complete. Train total: {total_train}, Val total: {total_val}, Combined: {total_train + total_val}")

if __name__ == "__main__":
    prepare_splits()
