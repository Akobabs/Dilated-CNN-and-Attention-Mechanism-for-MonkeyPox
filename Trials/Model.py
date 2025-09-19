# Step 2.1: Imports & Setup (run once)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Assume model class from before (DilatedAttentionResNet with num_classes=7 or 6 post-merge)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 2.2: Transforms (tailored for skin lesions)
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Handle lighting var
    transforms.RandomRotation(10),  # Slight for pose
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_val_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 2.3: Load Datasets
train_dataset = datasets.ImageFolder(base_path / 'train', transform=transform_train)
val_dataset = datasets.ImageFolder(base_path / 'val', transform=transform_val_test)
test_dataset = datasets.ImageFolder(base_path / 'test', transform=transform_val_test)

print(f"Train samples: {len(train_dataset)}, Classes: {train_dataset.classes}")  # Should match your classes

# Step 2.4: Handle Imbalance with Weighted Sampler
labels = np.array([label for _, label in train_dataset.samples])
class_weights = compute_class_weights('balanced', classes=np.unique(labels), y=labels)
sample_weights = class_weights[labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Step 2.5: Quick Test Load (verify)
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(f"Batch shape: {images.shape}, Labels: {labels[:5]}")  # Should be [32,3,224,224], tensor of class ids

# Step 2.6: Train Loop Snippet (integrate your model)
model = DilatedAttentionResNet(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Example epoch
for epoch in range(1):  # Test 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')