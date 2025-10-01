"""
🚗 Tire Quality Classification Model
Tire quality classification model using ResNet50
"""

import torch
import torch.nn as nn
import torchvision
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet50 feature extractor
    """
    def __init__(self, base_model):
        super(ResNetFeatureExtractor, self).__init__()
        # Remove the last layer (classifier) from ResNet50
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # Freeze pretrained weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Extract features from image
        
        Args:
            x: Input image (batch_size, 3, 224, 224)
            
        Returns:
            features: Extracted features (batch_size, 2048)
        """
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x


class FC_Classifier(nn.Module):
    """
    Custom fully connected classifier
    """
    def __init__(self, input_size=2048, dropout_rate=0.3):
        super(FC_Classifier, self).__init__()
        
        self.classifier = nn.Sequential(
            # First layer
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            
            # Second layer
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            
            # Third layer
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            
            # Fourth layer
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            
            # Fifth layer
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Classify features
        
        Args:
            x: Extracted features (batch_size, 2048)
            
        Returns:
            output: Classification result (batch_size, 1)
        """
        return self.classifier(x)


def load_pretrained_model():
    """
    Load pretrained model
    
    Returns:
        model: Loaded model
    """
    try:
        # Load ResNet50 pretrained on ImageNet
        base_resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        
        # Create feature extractor
        feature_extractor = ResNetFeatureExtractor(base_resnet).to(device)
        
        # Create classifier
        classifier = FC_Classifier().to(device)
        
        # Load trained weights
        model_path = "./best_models/best_model_acc0.9516_cpu.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            classifier.load_state_dict(state_dict)
            print(f"✅ Loaded pretrained model from {model_path}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
            print("Using untrained model")
        
        # Combine full model
        model = nn.Sequential(feature_extractor, classifier).to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


# Load model
try:
    model = load_pretrained_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    raise