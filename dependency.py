#!/usr/bin/env python3
"""
Test script to verify all dependencies are working
"""

def test_imports():
    success_count = 0
    total_tests = 6
    
    print("Testing library imports...\n")
    
    # Test NumPy
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        # Test basic operation
        arr = np.array([1, 2, 3])
        result = np.sum(arr)
        print(f"   - Basic operation test: sum([1,2,3]) = {result}")
        success_count += 1
    except Exception as e:
        print(f"❌ NumPy failed: {e}")
    
    # Test OpenCV
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        # Test basic operation
        img = cv2.imread('nonexistent.jpg')  # This should return None, not crash
        print(f"   - Basic operation test: imread returned {type(img)}")
        success_count += 1
    except Exception as e:
        print(f"❌ OpenCV failed: {e}")
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        # Test basic operation
        tensor = torch.tensor([1, 2, 3])
        result = torch.sum(tensor)
        print(f"   - Basic operation test: tensor sum = {result}")
        success_count += 1
    except Exception as e:
        print(f"❌ PyTorch failed: {e}")
    
    # Test TorchVision
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
        success_count += 1
    except Exception as e:
        print(f"❌ TorchVision failed: {e}")
    
    # Test Flask
    try:
        from flask import Flask
        app = Flask(__name__)
        print("✅ Flask: imported successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ Flask failed: {e}")
    
    # Test Pandas
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
        # Test basic operation
        df = pd.DataFrame({'a': [1, 2, 3]})
        print(f"   - Basic operation test: DataFrame created with shape {df.shape}")
        success_count += 1
    except Exception as e:
        print(f"❌ Pandas failed: {e}")
    
    print(f"\nSummary: {success_count}/{total_tests} libraries working correctly")
    
    if success_count == total_tests:
        print("🎉 All dependencies are working! You can run the TV remote detection script.")
        return True
    else:
        print("⚠️  Some dependencies failed. Please reinstall the failed libraries.")
        return False

def test_yolo_model():
    """Test if we can load the YOLO model"""
    print("\nTesting YOLO model loading...")
    try:
        import torch
        print("Attempting to load YOLOv5 model...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        print("✅ YOLO model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ YOLO model loading failed: {e}")
        print("This might be due to internet connection or torch hub issues.")
        return False

if __name__ == "__main__":
    print("="*50)
    print("DEPENDENCY TEST FOR TV REMOTE DETECTION")
    print("="*50)
    
    # Test basic imports
    basic_success = test_imports()
    
    # Test YOLO model if basic imports work
    if basic_success:
        model_success = test_yolo_model()
        if model_success:
            print("\n🎉 ALL TESTS PASSED! Ready to run the detection system.")
        else:
            print("\n⚠️  Basic libraries work, but YOLO model loading failed.")
            print("You might have internet connectivity issues.")
    else:
        print("\n❌ Basic library tests failed. Please reinstall dependencies.")
    
    print("\n" + "="*50)
