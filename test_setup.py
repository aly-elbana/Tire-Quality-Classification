"""
Test script to verify the project setup works correctly
Run this after installing dependencies to check everything is working
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision import failed: {e}")
        return False
    
    try:
        import gradio as gr
        print(f"✅ Gradio {gr.__version__}")
    except ImportError as e:
        print(f"❌ Gradio import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL (Pillow)")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the model can be loaded"""
    print("\n🔍 Testing model loading...")
    
    try:
        from model import model
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "app.py",
        "model.py", 
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]
    
    required_dirs = [
        "best_models",
        "tire_images_test"
    ]
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
            all_good = False
    
    for dir in required_dirs:
        if os.path.exists(dir):
            print(f"✅ {dir}/")
        else:
            print(f"❌ {dir}/ missing")
            all_good = False
    
    return all_good

def test_model_files():
    """Test if model files exist"""
    print("\n🔍 Testing model files...")
    
    model_dir = "best_models"
    if not os.path.exists(model_dir):
        print(f"❌ {model_dir} directory not found")
        return False
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if model_files:
        print(f"✅ Found {len(model_files)} model files:")
        for f in model_files:
            print(f"   - {f}")
        return True
    else:
        print("⚠️ No .pth model files found")
        return False

def main():
    """Run all tests"""
    print("🚗 Tire Quality Classification - Setup Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Model Files", test_model_files),
        ("Package Imports", test_imports),
        ("Model Loading", test_model_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The project is ready to use.")
        print("Run 'python app.py' to start the application.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        print("Make sure to install dependencies: pip install -r requirements.txt")
    
    return all_passed

if __name__ == "__main__":
    main()
