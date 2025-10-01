# ============================================================
# 📌 Section 1: Import Required Libraries
# ============================================================
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from PIL import Image
import datetime
import os

# ============================================================
# 📌 Section 2: Device Configuration + Speed Optimizations
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.mps.is_available():
    device = torch.device("mps")
    
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
        return Image.new("RGB", (224, 224))  # dummy fallback

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
# 📌 Section 7: Load Pretrained ResNet50 (Feature Extractor)
# ============================================================
base_resNet_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
)

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, _base_resNet_model):
        super().__init__()
        self.feature_extractor = nn.Sequential(*list(_base_resNet_model.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x

resNet_model = ResNetFeatureExtractor(base_resNet_model).to(device)

# Freeze pretrained weights
for param in resNet_model.parameters():
    param.requires_grad = False

resNet_model.eval()

# ============================================================
# 📌 Section 8: Define Custom Fully Connected (FC) Classifier
# ============================================================
class FC_Classifier(nn.Module):
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


# ============================================================
# 📌 Section 9: Combine ResNet + Custom Classifier
# ============================================================
fc_model = FC_Classifier().to(device)
model = nn.Sequential(resNet_model, fc_model).to(device)


# ============================================================
# 📌 Section 10: Define Optimizer + Loss + AMP Scaler + Checkpoints
# ============================================================
optimizer = torch.optim.AdamW(fc_model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()
epochs = 20
scaler = torch.amp.GradScaler("cuda")

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

if not os.path.exists("best_models"):
    os.mkdir("best_models")


# ============================================================
# 📌 Section 11: Training & Validation Loop with Early Stopping
# ============================================================
if __name__ == "__main__":
    print(f"Using device: {device}")

    patience = 5
    best_acc = 0.0
    patience_counter = 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(epochs):
        fc_model.train()
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

        val_acc = correct / len(val_dataset)
        print(f"Epoch {epoch+1}/{epochs} "
              f"| Train Loss={train_loss/len(train_loader):.4f} "
              f"| Val Loss={val_loss/len(val_loader):.4f} "
              f"| Val Acc={val_acc:.4f}")

        # Save checkpoint every epoch
        torch.save(fc_model.state_dict(), f"checkpoints/fc_classifier_epoch{epoch+1}.pth")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            best_model_path = f"best_models/best_model_{timestamp}_acc{best_acc:.4f}.pth"
            torch.save(fc_model.state_dict(), best_model_path)
            print(f"✅ New Best Model Saved: {best_model_path}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print("⛔ Early stopping triggered!")
            break
