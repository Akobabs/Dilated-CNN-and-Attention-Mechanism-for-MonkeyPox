import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import torchvision

# Define DilatedAttentionBlock (from your code)
class DilatedAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=2):
        super(DilatedAttentionBlock, self).__init__()
        reduction = max(16, out_channels // 16)
        self.dilated_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=dilation_rate, 
            dilation=dilation_rate, bias=False
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dilated_conv(x)
        out = self.bn(out)
        ca = self.channel_attention(out)
        out = out * ca
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        out = out * sa
        return self.relu(out)

# Define DilatedAttentionResNet (updated for channel matching)
class DilatedAttentionResNet(nn.Module):
    def __init__(self, num_classes=6, pretrained=False, backbone='resnet50'):
        super(DilatedAttentionResNet, self).__init__()
        if backbone == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained=pretrained)
            in_channels = 512
            d1_out = 256  # Adjusted for ResNet-18
            d2_out = 128
            d3_out = 64
            linear_in = 64
        elif backbone == 'resnet50':
            resnet = torchvision.models.resnet50(pretrained=pretrained)
            in_channels = 2048
            d1_out = 512
            d2_out = 256
            d3_out = 128
            linear_in = 128
        else:
            raise ValueError("Unsupported backbone. Use 'resnet18' or 'resnet50'.")
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.dilated_block1 = DilatedAttentionBlock(in_channels, d1_out, dilation_rate=2)
        self.dilated_block2 = DilatedAttentionBlock(d1_out, d2_out, dilation_rate=4)
        self.dilated_block3 = DilatedAttentionBlock(d2_out, d3_out, dilation_rate=8)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(linear_in, linear_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(linear_in // 2, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        x = self.dilated_block1(features)
        x = self.dilated_block2(x)
        x = self.dilated_block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Validation transform (from your code)
def get_validation_augmentation():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# Load model function with auto backbone and prefix stripping
@st.cache_resource
def load_model(model_path):
    try:
        # Auto-detect backbone from path
        if 'version 1' in model_path.lower() or 'version1' in model_path.lower():
            backbone = 'resnet18'
        elif 'version 2' in model_path.lower() or 'version2' in model_path.lower():
            backbone = 'resnet50'
        else:
            backbone = 'resnet50'  # Default
        checkpoint = torch.load(model_path, map_location='cpu')
        # Strip '_orig_mod.' prefix if present (for compiled models)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model = DilatedAttentionResNet(
            num_classes=checkpoint.get('num_classes', 6),
            pretrained=False,
            backbone=backbone
        )
        model.load_state_dict(new_state_dict)
        model.eval()
        class_names = checkpoint.get('class_names', ['Chickenpox', 'Cowpox', 'HFMD', 'Measles', 'Monkeypox', 'Normal'])
        val_acc = checkpoint.get('val_acc', 0.0)
        return model, class_names, val_acc, backbone
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        st.write("Path-based backbone detection failed. Ensure folders are named 'version1' or 'version2'.")
        return None, None, None, None

# Predict function for batch
def predict_batch(model, images, class_names, device='cpu'):
    transform = get_validation_augmentation()
    predictions = []
    
    for img in images:
        image_np = np.array(img.convert('RGB'))
        augmented = transform(image=image_np)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predictions.append({
            'predicted_class': class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': {class_names[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        })
    
    return predictions

# Streamlit App
st.title("Monkeypox Batch Classification Frontend")

st.write("""
Upload multiple skin lesion images for classification. Compare predictions across all saved models.
The best model (highest validation accuracy) is highlighted separately.
Backbone is auto-detected: 'resnet18' for version1 models, 'resnet50' for version2 models.
Compiled models ('_orig_mod.' prefix) are automatically handled.
""")

# Model directory input
model_dir = st.text_input("Model Directory Path", "./models/")
if not os.path.exists(model_dir):
    st.error(f"Model directory {model_dir} does not exist. Please create it and add .pth files.")
    st.stop()

model_files = [os.path.join(root, f) for root, dirs, files in os.walk(model_dir) for f in files if f.endswith('.pth')]
if not model_files:
    st.warning(f"No .pth files found in {model_dir} or subfolders. Please add model files.")
    st.stop()

# Load all models with auto backbone detection
models = []
best_model = None
best_val_acc = -1
best_model_name = None

for model_path in model_files:
    model, class_names, val_acc, backbone_used = load_model(model_path)
    if model is not None:
        models.append((model_path, model, class_names, val_acc, backbone_used))
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_model_name = model_path

st.write(f"Loaded {len(models)} models. Best model: {best_model_name} (Val Acc: {best_val_acc:.2f}%)")

# Image upload
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Optional ground truth CSV
gt_file = st.file_uploader("Upload ground truth CSV (optional, columns: image_path, true_label)", type=["csv"])

if uploaded_files:
    images = [Image.open(file) for file in uploaded_files]
    image_names = [file.name for file in uploaded_files]
    st.write(f"Processing {len(images)} images...")

    # Display images
    cols = st.columns(3)
    for i, img in enumerate(images):
        with cols[i % 3]:
            st.image(img, caption=image_names[i], use_column_width=True)

    # Batch predictions
    all_results = []
    for model_path, model, class_names, val_acc, backbone_used in models:
        predictions = predict_batch(model, images, class_names)
        for i, pred in enumerate(predictions):
            all_results.append({
                'Image': image_names[i],
                'Model': os.path.basename(model_path),
                'Predicted Class': pred['predicted_class'],
                'Confidence': f"{pred['confidence']:.2%}",
                'Val Acc': f"{val_acc:.2f}%" if val_acc is not None else 'N/A',
                'Backbone': backbone_used
            })

    # Best model predictions
    if best_model:
        st.subheader("Best Model Predictions")
        best_predictions = predict_batch(best_model, images, class_names)
        for i, pred in enumerate(best_predictions):
            st.write(f"**{image_names[i]} (Best Model: {os.path.basename(best_model_name)}, Val Acc: {best_val_acc:.2f}%)**")
            st.write(f"Predicted Class: {pred['predicted_class']}, Confidence: {pred['confidence']:.2%}")
            prob_df = pd.DataFrame({
                'Class': list(pred['all_probabilities'].keys()),
                'Probability': [f"{p:.2%}" for p in pred['all_probabilities'].values()]
            }).sort_values('Probability', ascending=False)
            st.dataframe(prob_df.style.background_gradient(cmap='viridis'))

    # Results table
    st.subheader("All Model Predictions")
    results_df = pd.DataFrame(all_results)
    st.dataframe(results_df.style.highlight_max(subset=['Val Acc'], color='lightgreen'))

    # Optional: Confusion matrix if ground truth provided
    if gt_file:
        gt_df = pd.read_csv(gt_file)
        if 'image_path' in gt_df.columns and 'true_label' in gt_df.columns:
            true_labels = []
            pred_labels = []
            for i, filename in enumerate(image_names):
                matching_row = gt_df[gt_df['image_path'].str.contains(filename, case=False)]
                if not matching_row.empty:
                    true_label = matching_row['true_label'].iloc[0]
                    pred = best_predictions[i]['predicted_class'] if best_model else predictions[i]['predicted_class']
                    if true_label in class_names:
                        true_labels.append(class_names.index(true_label))
                        pred_labels.append(class_names.index(pred))
            
            if true_labels:
                cm = confusion_matrix(true_labels, pred_labels)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names, yticklabels=class_names)
                plt.title('Confusion Matrix (Best Model)')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                st.pyplot(plt)
            else:
                st.warning("No matching ground truth labels found.")
        else:
            st.error("CSV must have 'image_path' and 'true_label' columns.")