import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import os
from pathlib import Path

# Define paths
base_path = Path('./monkeypox_dataset')
classes = ['Chickenpox', 'Cowpox', 'Healthy', 'HFMD', 'Measles', 'Monkeypox', 'Normal']

# Class-specific augmentations (heavier for minority classes)
minority_classes = ['Cowpox', 'Normal', 'Measles']  # <1000 images
train_augment_minority = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7),  # More rotation for diversity
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.7),  # Stronger for variability
    A.ElasticTransform(alpha=50, sigma=5, p=0.5),  # More frequent for lesion shapes
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

train_augment_majority = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
    A.ElasticTransform(alpha=50, sigma=5, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_test_augment = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Custom Dataset for Albumentations
class AlbumentationsDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        # Assign class-specific transforms
        self.is_minority = [cls in minority_classes for cls in self.classes]

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = np.array(Image.open(path).convert('RGB'))
        # Apply minority or majority transform based on class
        transform = train_augment_minority if self.is_minority[target] else train_augment_majority
        if self.transform:
            transform = self.transform  # Override for val/test
        augmented = transform(image=image)
        return augmented['image'], target

# Load datasets
train_dataset = AlbumentationsDataset(base_path / 'train', transform=None)  # Transform applied in __getitem__
val_dataset = AlbumentationsDataset(base_path / 'val', transform=val_test_augment)
test_dataset = AlbumentationsDataset(base_path / 'test', transform=val_test_augment)

# Class weights from your counts
class_counts = [982, 660, 1140, 1610, 732, 4398, 586]  # Chickenpox, Cowpox, Healthy, HFMD, Measles, Monkeypox, Normal
class_weights = compute_class_weight('balanced', classes=np.arange(len(classes)), y=np.array([label for _, label in train_dataset.samples]))
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Verify a batch
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(f"Batch shape: {images.shape}, Labels: {labels[:5]}")  # Should be [32, 3, 224, 224], tensor of class IDs





import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

# Define paths
base_path = './monkeypox_dataset'

# Albumentations for advanced transforms (ElasticTransform)
train_augment = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
    A.ElasticTransform(alpha=50, sigma=5, p=0.3),  # Simulates lesion deformation
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_test_augment = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Custom Dataset to use Albumentations with ImageFolder
class AlbumentationsDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = np.array(Image.open(path).convert('RGB'))
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, target

# Load datasets
train_dataset = AlbumentationsDataset(base_path + '/train', transform=train_augment)
val_dataset = AlbumentationsDataset(base_path + '/val', transform=val_test_augment)
test_dataset = AlbumentationsDataset(base_path + '/test', transform=val_test_augment)

# Class weights for imbalance (use counts from your cleaning)
class_counts = [284, 75, 55, 66, 161, 114, 114]  # Update with your df_counts (Monkeypox, Chickenpox, Measles, Cowpox, HFMD, Healthy, Normal)
class_weights = compute_class_weight('balanced', classes=np.arange(len(class_counts)), y