import os
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Define paths
base_path = Path('./monkeypox_dataset')
splits = ['train', 'val', 'test']
classes = ['Chickenpox', 'Cowpox', 'Healthy', 'HFMD', 'Measles', 'Monkeypox', 'Normal']

# Step 1.1: Count images per class/split (check balance & empties)
counts = {}
for split in splits:
    split_path = base_path / split
    counts[split] = {}
    for cls in classes:
        # Count images for each extension separately and sum
        total_images = sum(len(list((split_path / cls).glob(ext))) for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'])
        counts[split][cls] = total_images
df_counts = pd.DataFrame(counts).T
print("Image counts per split/class:\n", df_counts)

# Plot distribution
df_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Dataset Distribution')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Total per class
total_per_class = df_counts.sum(axis=0)
print("Total images per class:\n", total_per_class)
print(f"Grand total: {total_per_class.sum()}")


#%pip install ImageHash
# Step 1.2: Check for duplicates/redundancy (e.g., Healthy vs. Normal)
# Compute image hashes for quick dup detection (using PIL)
from imagehash import average_hash  # pip install ImageHash if not installed (but in your env, assume available or skip)

def get_image_hashes(folder_path):
    hashes = []
    for img_path in folder_path.glob('*.jpg'):  # Add .png etc.
        try:
            img = Image.open(img_path)
            hashes.append(str(average_hash(img)))
        except Exception:
            print(f"Corrupt: {img_path}")
    return hashes

healthy_train_hashes = get_image_hashes(base_path / 'train' / 'Healthy')
normal_train_hashes = get_image_hashes(base_path / 'train' / 'Normal')
common_hashes = set(healthy_train_hashes) & set(normal_train_hashes)
print(f"Potential duplicates between Healthy/Normal in train: {len(common_hashes)}")

# If >10% overlap, merge: e.g., mv Healthy/* Normal/ && rm -r Healthy (do manually after inspection)

# Step 1.3: Remove corrupt images
def clean_folder(folder_path):
    corrupt = []
    for img_path in folder_path.glob('*.jpg'):
        try:
            with Image.open(img_path) as img:
                img.verify()  # Checks integrity
        except Exception:
            corrupt.append(img_path)
            img_path.unlink()  # Delete
    print(f"Removed {len(corrupt)} corrupt images from {folder_path}")

for split in splits:
    for cls in classes:
        clean_folder(base_path / split / cls)
print("Cleaning complete!")

# Step 1.4: Visualize samples (robust version from before)
def get_first_image(folder_path):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    for ext in extensions:
        images = list(folder_path.glob(ext))
        if images:
            return images[0]
    return None

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
plotted = 0
skipped = []

for row in [0, 1]:
    start_idx = row * 4
    for col, cls in enumerate(classes[start_idx:start_idx + 4]):
        if start_idx + col >= len(classes):  # Handle case with fewer than 7 classes
            axes[row, col].axis('off')
            continue
        folder = base_path / 'train' / cls
        img_path = get_first_image(folder)
        if img_path is None:
            print(f"Warning: No images found in {folder} (skipping)")
            skipped.append(cls)
            axes[row, col].text(0.5, 0.5, f'No images\nin {cls}', ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')
            continue
        try:
            img = Image.open(img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'{cls} (Train Sample)')
            axes[row, col].axis('off')
            plotted += 1
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            axes[row, col].text(0.5, 0.5, f'Error loading\n{cls}', ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')

plt.suptitle(f'Dataset Samples (Plotted: {plotted}/{len(classes)}, Skipped: {skipped})')
plt.tight_layout()
plt.show()