# Tire Quality Classification System

AI-powered tire quality classification using deep learning

## Dataset
https://www.kaggle.com/datasets/warcoder/tyre-quality-classification

## Overview

This project aims to classify tire quality (good/bad) using deep learning techniques. The model uses ResNet50 as a feature extractor with a custom classifier to achieve high accuracy in classification.

## Features

- **High Accuracy**: Model achieves 95.16% accuracy in tire classification
- **Optimized Performance**: Uses ResNet50 with performance optimizations
- **Easy Interface**: Gradio application for model interaction
- **Advanced Training**: Supports Early Stopping and AMP
- **Image Processing**: Automatic image enhancement support

## Project Structure

```
├── app.py                # Main Gradio application
├── main.py               # Main training script
├── cnn.main.py           # CNN training script
├── model.py              # Model definition
├── model_testing.py      # Model testing
├── model.testing.py      # Alternative testing
├── best_models/          # Saved best models
├── tire_images_test/     # Test images
└── requirements.txt      # Project requirements
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone URL
cd tire-classification
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

## Troubleshooting

### Common Issues

**1. Model Loading Error:**

- The app will automatically try to load the best available model
- If no model is found, it will use an untrained model
- Check that `best_models/` directory contains `.pth` files

**2. CUDA/GPU Issues:**

- The app automatically detects and uses available hardware
- CPU fallback is always available
- No GPU required for basic functionality

**3. Missing Dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Port Already in Use:**

- Change port in `app.py`: `demo.launch(server_port=7861)`
- Or kill existing process: `lsof -ti:7860 | xargs kill -9`

## Trained Models

### Main Model (ResNet50 + FC Classifier)

- **Architecture**: ResNet50 (frozen) + Custom FC Layers
- **Accuracy**: 95.16%
- **Input Size**: 224x224x3
- **Output**: Binary Classification (Good/Bad Tire)

### Alternative Model (CNN-based)

- **Architecture**: ResNet50 + Custom CNN Classifier
- **Features**: Adaptive Pooling, Batch Normalization
- **Performance**: Optimized for different image sizes

## Performance Results

  ResNet50 + FC  =>     95.16%   
  ResNet50 + CNN  =>    64.82%    
## Usage

### Training

```python
# Train main model
python main.py

# Train CNN model
python cnn.main.py
```

### Testing

```python
# Test the model
python model.testing.py
```

### Interactive Application

```python
# Run Gradio interface
python app.py
```

## Data Structure

```
images/
├── good/          # Good tire images
│   ├── tire1.jpg
│   ├── tire2.jpg
│   └── ...
└── bad/           # Bad tire images
    ├── tire1.jpg
    ├── tire2.jpg
    └── ...
```

## Advanced Settings

### Training Parameters

- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Epochs**: 20
- **Optimizer**: AdamW
- **Loss Function**: BCEWithLogitsLoss

### Performance Optimizations

- **Mixed Precision Training**: Enabled
- **Data Augmentation**: Enabled
- **Early Stopping**: Enabled (patience=5)
- **GPU Support**: CUDA/MPS

## API Usage

```python
from model import model
from PIL import Image
import torch
import torchvision.transforms as transforms


# Image Loading
image = Image.open("tire_image.jpg")

# Process image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Make prediction
model.eval()
with torch.no_grad():
    input_tensor = transform(image).unsqueeze(0)
    prediction = torch.sigmoid(model(input_tensor))

    if prediction > 0.5:
        print(f"Good tire: {prediction.item()*100:.2f}%")
    else:
        print(f"Bad tire: {(1-prediction.item())*100:.2f}%")
```
