"""
🧪 Tire Quality Model Testing Script
Tire quality classification model testing script
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}")

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

class ResNetFeatureExtractor(nn.Module):
    """ResNet50 feature extractor"""
    def __init__(self, base_model):
        super().__init__()
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x

class FC_Classifier(nn.Module):
    """Fully connected classifier"""
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
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.drop3(self.relu3(self.fc3(x)))
        x = self.drop4(self.relu4(self.fc4(x)))
        x = self.drop5(self.relu5(self.fc5(x)))
        x = self.fc_out(x)
        return x

def load_model():
    """Load trained model"""
    try:
        # Load ResNet50
        base_resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        
        # Create model
        resnet_model = ResNetFeatureExtractor(base_resnet).to(device)
        fc_model = FC_Classifier().to(device)
        
        # Load trained weights
        model_path = "./best_models/best_model_acc0.9516_cpu.pth"
        if os.path.exists(model_path):
            fc_state_dict = torch.load(model_path, map_location=device)
            fc_model.load_state_dict(fc_state_dict)
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
        
        # Combine model
        model = nn.Sequential(resnet_model, fc_model).to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def test_single_image(model, image_path, image_name):
    """
    Test single image
    
    Args:
        model: Loaded model
        image_path: Image path
        image_name: Image name
        
    Returns:
        tuple: (prediction, confidence, result_text)
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Process image
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.sigmoid(output).item()
        
        # Determine result
        if prediction > 0.5:
            confidence = prediction * 100
            result_text = f"🟢 Good Tire: {confidence:.2f}%"
        else:
            confidence = (1 - prediction) * 100
            result_text = f"🔴 Bad Tire: {confidence:.2f}%"
        
        return prediction, confidence, result_text
        
    except Exception as e:
        print(f"Error testing {image_name}: {str(e)}")
        return None, None, f"❌ Error processing {image_name}"

def run_tests():
    """Run all tests"""
    print("🚀 Starting tire quality testing...")
    
    # Load model
    model = load_model()
    
    # Test images directory
    test_dir = Path("./tire_images_test")
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in test_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ No image files found in {test_dir}")
        return
    
    print(f"📸 Found {len(image_files)} test images")
    
    # Test results
    results = []
    
    # Test each image
    for image_file in sorted(image_files):
        prediction, confidence, result_text = test_single_image(
            model, image_file, image_file.name
        )
        
        if prediction is not None:
            results.append({
                'name': image_file.name,
                'prediction': prediction,
                'confidence': confidence,
                'result': result_text
            })
            print(f"{image_file.name}: {result_text}")
        else:
            print(f"{image_file.name}: ❌ Processing failed")
    
    # Results statistics
    if results:
        good_count = sum(1 for r in results if r['prediction'] > 0.5)
        bad_count = len(results) - good_count
        
        print(f"\n📊 Results Statistics:")
        print(f"🟢 Good tires: {good_count}")
        print(f"🔴 Bad tires: {bad_count}")
        print(f"📈 Total images: {len(results)}")
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"📊 Average confidence: {avg_confidence:.2f}%")
    
    print("✅ Testing completed!")

if __name__ == "__main__":
    run_tests()