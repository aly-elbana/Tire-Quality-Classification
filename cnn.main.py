# ============================================================
# 📌 Section 1: Import Required Libraries
# ============================================================
import torch
import torch.nn as nn
import torchvision
import datetime
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import random_split, DataLoader

# ============================================================
# 📌 Section 2: Device Configuration + Speed Optimizations
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 📌 Section 3: Safe Loader for Corrupted Images
# ============================================================
def pil_loader_safe(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except Exception:
        print(f"⚠️ Skipping corrupted image: {path}")
        return Image.new("RGB", (224, 224))
    
# ============================================================
# 📌 Section 4: Data Preprocessing (with Augmentation)
# ============================================================
preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# 📌 Section 5: Load Dataset
# ============================================================
dataset = torchvision.datasets.ImageFolder(
    root="./images",
    transform=preprocess,
    loader=pil_loader_safe
)

# ============================================================
# 📌 Section 6: Split Dataset into Training and Validation
# ============================================================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset,
                          batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True)

val_loader = DataLoader(val_dataset,
                        batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

# ============================================================
# 📌 Section 7: Define ResNet Feature Extractor
# ============================================================
base_resNet_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
)

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, _base_resNet_model):
        super(ResNetFeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(*list(_base_resNet_model.children())[:-1])
        
    def forward(self, x):
        x = self.feature_extractor(x)
        return x
    
resNet_model = ResNetFeatureExtractor(base_resNet_model).to(device)

for param in resNet_model.parameters():
    param.requires_grad = False
    
resNet_model.eval()

# ============================================================
# 📌 Section 8: Define Custom Fully Connected (FC) Classifier
# ============================================================
class External_Classifier(nn.Module):
    def __init__(self):
        super(External_Classifier, self).__init__()
        self.l1 = nn.Conv2d(2048, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.p1 = nn.AdaptiveAvgPool2d((3, 3))   # Instead of MaxPool: stable 3x3

        self.l2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.p2 = nn.AdaptiveAvgPool2d((1, 1))   # Instead of MaxPool: stable 1x1

        self.l3 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.fc_out = nn.Linear(256, 1)

    def forward(self, x):
        # Debug shapes
        # print("Input:", x.shape)
        x = self.p1(torch.relu(self.bn1(self.l1(x))))
        # print("After p1:", x.shape)
        x = self.p2(torch.relu(self.bn2(self.l2(x))))
        # print("After p2:", x.shape)
        x = torch.relu(self.bn3(self.l3(x)))
        # print("After l3:", x.shape)

        x = self.gap(x)
        # print("After GAP:", x.shape)
        x = torch.flatten(x, 1)
        x = self.fc_out(x)
        return x

# ============================================================
# 📌 Section 9: Combine ResNet + Custom Classifier
# ============================================================
classifier_model = External_Classifier().to(device)
model = nn.Sequential(resNet_model, classifier_model).to(device)

# ============================================================
# 📌 Section 10: Define Optimizer + Loss + AMP Scaler + Checkpoints
# ============================================================
optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()
epochs = 20
scaler = torch.amp.GradScaler("cuda")

if not os.path.exists("checkpoints_cnn"):
    os.mkdir("checkpoints_cnn")

if not os.path.exists("best_models_cnn"):
    os.mkdir("best_models_cnn")

# ============================================================
# 📌 Section 11: Training & Validation Loop with Early Stopping
# ============================================================
if __name__ == "__main__":
    print(f"Using device: {device}")
    
    patience = 5
    best_acc = 0.0
    patience_counter = 0
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    for epoch in range(epochs):
        classifier_model.train()
        train_loss = 0.0
        
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).reshape(-1, 1).float()
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(xb)
                loss = loss_fn(outputs, yb)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).reshape(-1, 1).float()
                
                with torch.amp.autocast("cuda"):
                    outputs = model(xb)
                    loss = loss_fn(outputs, yb)
                    
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == yb).sum().item()
            
        val_acc = correct / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} "
                f"| Train Loss={train_loss/len(train_loader):.4f} "
                f"| Val Loss={val_loss/len(val_loader):.4f} "
                f"| Val Acc={val_acc:.4f}")
        
        torch.save(classifier_model.state_dict(), f"checkpoints_cnn/classifier_model_epoch{epoch+1}.pth")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            best_model_path = f"best_models_cnn/best_model_{timestamp}_acc{best_acc:.4f}.pth"
            torch.save(classifier_model.state_dict(), best_model_path)
            print(f"✅ New Best Model Saved: {best_model_path}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print("⛔ Early stopping triggered!")
            break
