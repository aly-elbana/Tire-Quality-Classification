import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from model import model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.mps.is_available():
    device = torch.device("mps")
    
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



tire0 = Image.open("../Day10/tire_images_test/tire0.jpg")
tire1 = Image.open("../Day10/tire_images_test/tire1.jpg")
tire2 = Image.open("../Day10/tire_images_test/tire2.jpg")
tire3 = Image.open("../Day10/tire_images_test/tire3.jpg")
tire4 = Image.open("../Day10/tire_images_test/tire4.jpg")
tire5 = Image.open("../Day10/tire_images_test/tire5.jpg")
tire6 = Image.open("../Day10/tire_images_test/tire6.jpg")
tire7 = Image.open("../Day10/tire_images_test/tire7.png")
tire8 = Image.open("../Day10/tire_images_test/tire8.png")
tire9 = Image.open("../Day10/tire_images_test/tire9.jpg")
tire10 = Image.open("../Day10/tire_images_test/tire10.jpg")
tire11 = Image.open("../Day10/tire_images_test/tire11.jpg")
tire12 = Image.open("../Day10/tire_images_test/tire12.jpg")
tire13 = Image.open("../Day10/tire_images_test/tire13.jpg")

tire0 = preprocess(tire0).unsqueeze(0).to(device)
tire1 = preprocess(tire1).unsqueeze(0).to(device)
tire2 = preprocess(tire2).unsqueeze(0).to(device)
tire3 = preprocess(tire3).unsqueeze(0).to(device)
tire4 = preprocess(tire4).unsqueeze(0).to(device)
tire5 = preprocess(tire5).unsqueeze(0).to(device)
tire6 = preprocess(tire6).unsqueeze(0).to(device)
tire7 = preprocess(tire7).unsqueeze(0).to(device)
tire8 = preprocess(tire8).unsqueeze(0).to(device)
tire9 = preprocess(tire9).unsqueeze(0).to(device)
tire10 = preprocess(tire10).unsqueeze(0).to(device)
tire11 = preprocess(tire11).unsqueeze(0).to(device)
tire12 = preprocess(tire12).unsqueeze(0).to(device)
tire13 = preprocess(tire13).unsqueeze(0).to(device)

pred_list = []

model.eval()
with torch.no_grad():
    y_pred = torch.sigmoid(model(tire0))
    print(y_pred)
    pred_list.append(y_pred)
    
    y_pred = torch.sigmoid(model(tire1))
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire2))
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire3))
    print(y_pred)
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire4))
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire5))
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire6))
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire7))
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire8))
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire9))
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire10))
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire11))
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire12))
    pred_list.append(y_pred)

    y_pred = torch.sigmoid(model(tire13))
    pred_list.append(y_pred)

for i, pred in enumerate(pred_list):
    if pred > 0.5:
        print(f"tire{i} is {pred.item()*100:.2f}% good")
    else:
        print(f"tire{i} is {100.0 - pred.item()*100:.2f}% bad")