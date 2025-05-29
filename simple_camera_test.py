#!/usr/bin/env python3
import subprocess
import time
import numpy as np
import cv2
import os

def test_libcamera_direct():
    """Test direct libcamera capture"""
    print("Testing direct libcamera capture...")
    
    cmd = [
        'libcamera-vid',
        '--width', '640',
        '--height', '480',
        '--framerate', '10',
        '--timeout', '0',  # Infinite
        '--codec', 'yuv420',
        '--output', '-',
        '--nopreview'
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        frame_size = 640 * 480 * 3 // 2  # YUV420 format
        
        print("Reading frames...")
        success_count = 0
        
        for i in range(20):  # Try 20 frames
            try:
                raw_data = process.stdout.read(frame_size)
                if len(raw_data) == frame_size:
                    # Convert YUV to BGR
                    yuv_data = np.frombuffer(raw_data, dtype=np.uint8)
                    yuv_frame = yuv_data.reshape((480 * 3 // 2, 640))
                    bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                    
                    print(f"Frame {i+1}: {bgr_frame.shape}")
                    success_count += 1
                    
                    # Save first frame as test
                    if i == 0:
                        cv2.imwrite('test_frame.jpg', bgr_frame)
                        print("Saved test_frame.jpg")
                else:
                    print(f"Frame {i+1}: Wrong size {len(raw_data)}")
                    
            except Exception as e:
                print(f"Frame {i+1}: Error {e}")
        
        process.terminate()
        print(f"Success: {success_count}/20 frames")
        
        return success_count > 10
        
    except Exception as e:
        print(f"libcamera test failed: {e}")
        return False

def test_opencv_simple():
    """Test simple OpenCV capture"""
    print("\nTesting OpenCV capture...")
    
    # Try different methods
    methods = [
        ("Device 0 + V4L2", lambda: cv2.VideoCapture(0, cv2.CAP_V4L2)),
        ("Device 0 + Any", lambda: cv2.VideoCapture(0, cv2.CAP_ANY)),
        ("/dev/video0 + V4L2", lambda: cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)),
    ]
    
    for name, create_cap in methods:
        print(f"\nTrying {name}...")
        try:
            cap = create_cap()
            if cap.isOpened():
                print("  Camera opened")
                
                # Set properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 10)
                
                # Try to read frames
                success_count = 0
                for i in range(10):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        success_count += 1
                        if i == 0:
                            print(f"  First frame: {frame.shape}")
                    time.sleep(0.1)
                
                cap.release()
                print(f"  Success: {success_count}/10 frames")
                
                if success_count > 5:
                    return name
            else:
                print("  Cannot open camera")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    return None

def test_gstreamer_simple():
    """Test simple GStreamer pipeline"""
    print("\nTesting GStreamer...")
    
    pipeline = "libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! appsink drop=1"
    
    try:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("  GStreamer pipeline opened")
            
            success_count = 0
            for i in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    success_count += 1
                    if i == 0:
                        print(f"  First frame: {frame.shape}")
                time.sleep(0.2)
            
            cap.release()
            print(f"  Success: {success_count}/10 frames")
            return success_count > 5
        else:
            print("  Cannot open GStreamer pipeline")
            
    except Exception as e:
        print(f"  GStreamer error: {e}")
    
    return False

def main():
    print("=== SIMPLE CAMERA TEST ===")
    
    # Check basic requirements
    if not os.path.exists('/dev/video0'):
        print("ERROR: /dev/video0 not found")
        print("Run: sudo raspi-config -> Interface Options -> Camera -> Enable")
        return
    
    # Test methods
    libcamera_works = test_libcamera_direct()
    opencv_method = test_opencv_simple()
    gstreamer_works = test_gstreamer_simple()
    
    print("\n=== RESULTS ===")
    print(f"libcamera-vid: {'✓' if libcamera_works else '✗'}")
    print(f"OpenCV: {'✓ ' + opencv_method if opencv_method else '✗'}")
    print(f"GStreamer: {'✓' if gstreamer_works else '✗'}")
    
    if libcamera_works:
        print("\nRECOMMENDATION: Use libcamera-vid method in main code")
    elif gstreamer_works:
        print("\nRECOMMENDATION: Use GStreamer method in main code")
    elif opencv_method:
        print(f"\nRECOMMENDATION: Use {opencv_method} in main code")
    else:
        print("\nERROR: No working camera method found")
        print("Try: sudo python simple_camera_test.py")

if __name__ == "__main__":
    main()
