"""
Tire Quality Classification App
AI-powered tire quality classification application
"""

import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from model import model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def predict_tire_quality(image_pixels):
    """
    Classify tire quality
    
    Args:
        image_pixels: Tire image as numpy array
        
    Returns:
        str: Classification result with percentage
    """
    try:
        # Convert image to PIL Image
        img = Image.fromarray(image_pixels)
        
        # Process image
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.sigmoid(output).item()
        
        # Determine result
        if prediction < 0.5:
            confidence = round(100 - prediction * 100, 2)
            result = f"🔴 Bad Tire: {confidence}%"
        else:
            confidence = round(prediction * 100, 2)
            result = f"🟢 Good Tire: {confidence}%"
        
        print(f"Prediction: {prediction:.4f}, Result: {result}")
        return result
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return f"❌ Error processing image: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_tire_quality,
    inputs=gr.Image(
        label="📸 Upload Tire Image",
        type="numpy"
    ),
    outputs=gr.Textbox(
        label="🎯 Classification Result",
        lines=2
    ),
    title="🚗 Tire Quality Classification System",
    description="""
    **Welcome to the Tire Quality Classification System!**
    
    📋 **How to use:**
    1. Upload a clear tire image
    2. Wait for the result
    3. Get quality assessment with percentage
    
    ⚠️ **Note:** For best results, ensure the image is clear and well-lit
    """,
    examples=[
        ["tire_images_test/tire0.jpg"],
        ["tire_images_test/tire1.jpg"],
        ["tire_images_test/tire2.jpg"],
        ["tire_images_test/tire3.jpg"],
    ],
    cache_examples=True,
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    print("Starting Tire Classification App...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        show_error=True
    )
