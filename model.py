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
    def __init__(self):
        super(FC_Classifier, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        """
        Classify features
        
        Args:
            x: Extracted features (batch_size, 2048)
            
        Returns:
            output: Classification result (batch_size, 1)
        """
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.drop3(self.relu3(self.fc3(x)))
        x = self.drop4(self.relu4(self.fc4(x)))
        x = self.drop5(self.relu5(self.fc5(x)))
        x = self.fc_out(x)
        return x


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
        
        # Load trained weights - try multiple model files
        model_paths = [
            "./best_models/best_model_acc0.9516_cpu.pth",
            "./best_models/best_model_20250827_123421_acc0.9516.pth"
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    state_dict = torch.load(model_path, map_location=device)
                    classifier.load_state_dict(state_dict)
                    print(f"✅ Loaded pretrained model from {model_path} (95.16% accuracy)")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {model_path}: {str(e)}")
                    continue
        
        if not model_loaded:
            print("⚠️ No compatible model files found. Using untrained model.")
            print("Available models:", [f for f in os.listdir("./best_models/") if f.endswith('.pth')])
        
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