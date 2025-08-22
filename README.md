
# Custom CNN for SVHN

A PyTorch implementation of a custom Convolutional Neural Network (CNN) for digit recognition on the **SVHN (Street View House Numbers)** dataset.  

## Quick Start
```bash
# Recommended Python version 3.10+
pip install torch torchvision matplotlib seaborn scikit-learn tqdm
jupyter notebook custom_cnn_for_SVHN.ipynb
```

## How It Works
- **Dataset**: SVHN (Street View House Numbers)  
  - ~80% for training, ~20% for validation, plus test set.  
  - Images are normalized and converted to tensors.  

- **Model (JustCNN)**:
  - 4 convolutional layers with ReLU + MaxPooling:
    - Conv2D: 3→64  
    - Conv2D: 64→128  
    - Conv2D: 128→256  
    - Conv2D: 256→512  
  - Fully connected layers: 4096 → 512 → 256 → 128 → 10  
  - Output: 10 classes (digits `0–9`)  

- **Training**:  
  - Optimizer: Adam  
  - Loss: CrossEntropyLoss  
  - Batch size: 64  
  - Accuracy and loss tracked per epoch  

- **Validation**:
  - Classification Report (precision, recall, f1-score)  
  - Confusion Matrix (visualized with seaborn)  
  - ROC AUC Score (multiclass OVR)  

## Results
- **Validation Accuracy**: ~0.897  
- **ROC AUC Score**: ~0.955  
- Classification performance is balanced across all 10 digits.  
- Clear separation between classes shown in the confusion matrix.  

## Dependencies
- `torch`, `torchvision`  
- `numpy`, `matplotlib`, `seaborn`  
- `scikit-learn`  
- `tqdm`  
