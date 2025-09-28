# Monkeypox Classification Project

This project implements a deep learning classification system for detecting skin lesions caused by Monkeypox and related diseases. It uses **ResNet-based architectures with Dilated Attention Blocks** and includes Jupyter training notebooks and a **Streamlit web application** for batch inference.

Here’s a tighter, clearer refactor of your dataset section, keeping it professional and research-ready:

---

## Dataset

The dataset used in this project is a unified collection of skin lesion images for classifying six categories: **Chickenpox, Cowpox, Hand-Foot-and-Mouth Disease (HFMD), Measles, Monkeypox, and Normal (healthy skin)**. It was created by combining and standardizing multiple publicly available datasets to ensure consistent quality, resolution, and labeling for deep learning tasks.

The unified dataset is hosted on Google Drive: [Download Dataset](https://drive.google.com/file/d/1zpnQR_IBRXO68u6qI8i_nnuXTnkNHvlk/view?usp=sharing).

### Dataset Description

* **Classes**: 6 (Chickenpox, Cowpox, HFMD, Measles, Monkeypox, Normal)
* **Images**: Thousands of samples (exact count depends on the data split), with balanced representation across classes
* **Format**: JPG/PNG, resized to 224×224 pixels for ResNet-based models
* **Splits**: Organized into training, validation, and test sets, with CSV metadata files containing image paths and labels
* **Preprocessing**:

  * Normalized using ImageNet mean and standard deviation (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
  * Augmentation applied during training to improve generalization

### Sources

The unified dataset was constructed by refactoring and merging images from the following open-access sources:

* **Monkeypox Skin Lesion Dataset** – Yousry, A., et al. (2022), Kaggle
* **Skin lesion datasets for related diseases** – Public repositories such as DermNet and ISIC Archive
* **Additional open-access databases** – Curated images for Chickenpox, Cowpox, HFMD, Measles, and Normal skin, annotated for consistency

### Citation

When using this dataset, please cite the original dataset creators:

* Yousry, A., et al. (2022). *Monkeypox Skin Lesion Dataset*. Kaggle.
* Other sources as referenced in their respective repositories.


## Project Structure

* **Jupyter Notebooks**

  * `type_1.ipynb`: ResNet-18 backbone with Dilated Attention Blocks (dilations 2/4/8, channel & spatial attention).
  * `type_2.ipynb`: ResNet-50 backbone with Dilated Attention Blocks, optimized with Kaiming initialization.
  * `type_2.1.ipynb`, `type_2.2.ipynb`, `type_2.3.ipynb`: Variants of ResNet-50 experiments.
  * `type_3.ipynb`: Additional architecture experiments.
  * `trials/`: Early prototype experiments.

* **Streamlit App**

  * `app.py`: Web application for batch image classification.
    Features:

    * Auto-detects backbone (ResNet-18 for version1, ResNet-50 for version2).
    * Supports models with `_orig_mod.` prefixes.
    * Displays predictions from all models, highlights best model by validation accuracy.
    * Optional ground truth CSV upload for confusion matrix visualization.

* **Model Checkpoints**

  * `./models/version1/`: ResNet-18 models (e.g., `version 1-100 epoch.pth`).
  * `./models/version2/`: ResNet-50 models (e.g., `version 2-100 epoch.pth`, `version 2-30 epoch.pth`, `version 2-50 epoch.pth`).

## Installation

```bash
git clone https://github.com/Akobabs/Dilated-CNN-and-Attention-Mechanism-for-MonkeyPox.git
cd Dilated-CNN-and-Attention-Mechanism-for-MonkeyPox
pip install torch torchvision albumentations pandas numpy pillow matplotlib seaborn scikit-learn streamlit
```

Download the dataset and place it under `./data/`.

## Usage

### Training

1. Open a notebook (`type_1.ipynb`, `type_2.ipynb`, etc.) in Jupyter Lab/Notebook.
2. Update dataset path to match your setup.
3. Run all cells to train. Models are saved in `./models/version1/` or `./models/version2/`.

**Checkpoint Saving Example:**

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'val_acc': val_acc,
    'val_loss': val_loss,
    'class_names': self.class_names,
    'backbone': 'resnet18' if 'resnet18' in str(self.model.backbone) else 'resnet50',
    'num_classes': len(self.class_names)
}, 'best_monkeypox_model.pth')
```

### Inference with Streamlit

```bash
streamlit run app.py
```

In the app:

* Set model directory (`./models/` by default).
* Upload one or more images (JPG, JPEG, PNG).
* Optionally upload CSV (`image_path,true_label`) for confusion matrix.
* View predictions from all models and detailed output from the best one.

### Local Inference (Without Streamlit)

A test script is provided:

```bash
python test_inference_local.py
```

It loads a trained checkpoint, applies validation transforms, and outputs predictions with probabilities.

## Architecture Details

* **Type 1 (ResNet-18)**

  * Backbone: ResNet-18, 512 output channels.
  * Dilated Attention Blocks: 3 (dilations 2, 4, 8).
  * Classifier: Dropout → Linear(64→32) → ReLU → Dropout → Linear(32→6).

* **Type 2 (ResNet-50)**

  * Backbone: ResNet-50, 2048 output channels.
  * Dilated Attention Blocks: 3 (dilations 2, 4, 8).
  * Classifier: Dropout → Linear(128→64) → ReLU → Dropout → Linear(64→6).
  * Includes Kaiming initialization.

* **Trials**: Exploratory models before finalizing Type 1 and Type 2.

## Requirements

* Python 3.8+
* PyTorch, Torchvision
* Albumentations
* Streamlit
* Pandas, NumPy
* Pillow, Matplotlib, Seaborn
* Scikit-learn

## Contributing

1. Fork the repo.
2. Create a feature branch:

   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:

   ```bash
   git commit -m "Add feature"
   ```
4. Push to the branch:

   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License

This project is licensed under the **MIT License**.

---