#!/usr/bin/env python3
import subprocess
import os
import sys
import time
import cv2

def run_command(cmd, timeout=10):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_camera_detection():
    """Check if camera is detected by the system"""
    print("=== 1. CAMERA DETECTION ===")
    
    # Check libcamera detection
    success, stdout, stderr = run_command("libcamera-hello --list-cameras")
    if success and "Available cameras" in stdout:
        print("✓ libcamera detects camera:")
        print(stdout)
    else:
        print("✗ libcamera detection failed:")
        print(f"Error: {stderr}")
    
    # Check video devices
    print("\n--- Video Devices ---")
    for i in range(5):
        device = f"/dev/video{i}"
        if os.path.exists(device):
            print(f"✓ {device} exists")
            
            # Get device info
            success, stdout, stderr = run_command(f"v4l2-ctl --info -d {device}")
            if success:
                print(f"  Info: {stdout.split('Card type')[1].split('Bus info')[0].strip()}" if 'Card type' in stdout else "  No card info")
    
    # Check camera module in boot config
    print("\n--- Boot Configuration ---")
    if os.path.exists("/boot/config.txt"):
        with open("/boot/config.txt", "r") as f:
            config = f.read()
            if "camera_auto_detect=1" in config:
                print("✓ camera_auto_detect=1 found in config.txt")
            elif "dtoverlay=imx708" in config:
                print("✓ IMX708 overlay found in config.txt")
            else:
                print("? No camera configuration found in config.txt")

def check_permissions():
    """Check user permissions"""
    print("\n=== 2. PERMISSIONS ===")
    
    # Check user groups
    success, stdout, stderr = run_command("groups")
    if success:
        groups = stdout.strip().split()
        if "video" in groups:
            print("✓ User is in 'video' group")
        else:
            print("✗ User NOT in 'video' group")
            print("  Fix: sudo usermod -a -G video $USER")
        
        if "camera" in groups:
            print("✓ User is in 'camera' group")
    
    # Check device permissions
    for i in range(3):
        device = f"/dev/video{i}"
        if os.path.exists(device):
            stat = os.stat(device)
            perms = oct(stat.st_mode)[-3:]
            print(f"  {device} permissions: {perms}")

def test_libcamera_commands():
    """Test basic libcamera commands"""
    print("\n=== 3. LIBCAMERA COMMANDS ===")
    
    # Test libcamera-hello
    print("Testing libcamera-hello...")
    success, stdout, stderr = run_command("timeout 5 libcamera-hello --timeout 2000")
    if success:
        print("✓ libcamera-hello works")
    else:
        print("✗ libcamera-hello failed:")
        print(f"Error: {stderr}")
    
    # Test libcamera-still
    print("\nTesting libcamera-still...")
    success, stdout, stderr = run_command("libcamera-still -o /tmp/test.jpg --timeout 3000")
    if success and os.path.exists("/tmp/test.jpg"):
        print("✓ libcamera-still works")
        os.remove("/tmp/test.jpg")
    else:
        print("✗ libcamera-still failed:")
        print(f"Error: {stderr}")

def test_gstreamer():
    """Test GStreamer installation and pipelines"""
    print("\n=== 4. GSTREAMER ===")
    
    # Check GStreamer installation
    success, stdout, stderr = run_command("gst-inspect-1.0 libcamerasrc")
    if success:
        print("✓ GStreamer libcamerasrc plugin available")
    else:
        print("✗ GStreamer libcamerasrc plugin missing")
        print("  Install: sudo apt install gstreamer1.0-libcamera")
    
    # Test simple pipeline
    print("\nTesting GStreamer pipeline...")
    success, stdout, stderr = run_command("timeout 5 gst-launch-1.0 libcamerasrc num-buffers=10 ! fakesink")
    if success:
        print("✓ Basic GStreamer pipeline works")
    else:
        print("✗ GStreamer pipeline failed:")
        print(f"Error: {stderr}")

def test_opencv_backends():
    """Test OpenCV with different backends"""
    print("\n=== 5. OPENCV BACKENDS ===")
    
    backends = [
        ("V4L2", cv2.CAP_V4L2),
        ("GStreamer", cv2.CAP_GSTREAMER),
        ("Any", cv2.CAP_ANY)
    ]
    
    devices = [0, "/dev/video0", "/dev/video1"]
    
    for backend_name, backend in backends:
        print(f"\n--- Testing {backend_name} backend ---")
        
        for device in devices:
            try:
                print(f"Trying device {device}...")
                cap = cv2.VideoCapture(device, backend)
                
                if cap.isOpened():
                    # Try to read a frame
                    for attempt in range(5):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"✓ {backend_name} + {device} works! Frame: {frame.shape}")
                            cap.release()
                            return device, backend  # Return first working combination
                        time.sleep(0.2)
                    
                    print(f"  Device opened but no frames")
                    cap.release()
                else:
                    print(f"  Cannot open device {device}")
                    
            except Exception as e:
                print(f"  Error: {e}")
    
    return None, None

def test_specific_pipelines():
    """Test specific GStreamer pipelines"""
    print("\n=== 6. SPECIFIC GSTREAMER PIPELINES ===")
    
    pipelines = [
        "libcamerasrc ! video/x-raw,width=640,height=480,framerate=15/1 ! videoconvert ! appsink drop=1",
        "libcamerasrc ! videoconvert ! appsink drop=1",
        "libcamerasrc ! video/x-raw,format=RGB ! videoconvert ! appsink drop=1"
    ]
    
    for i, pipeline in enumerate(pipelines):
        print(f"\nTesting pipeline {i+1}:")
        print(f"  {pipeline}")
        
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print("  Pipeline opened successfully")
                
                # Try to read frames
                success_count = 0
                for attempt in range(10):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        success_count += 1
                    time.sleep(0.1)
                
                if success_count > 0:
                    print(f"✓ Pipeline {i+1} works! Got {success_count}/10 frames")
                    cap.release()
                    return pipeline
                else:
                    print(f"  Pipeline opened but no frames")
                
                cap.release()
            else:
                print(f"  Cannot open pipeline {i+1}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    return None

def main():
    print("RASPBERRY PI CAMERA DIAGNOSTIC")
    print("=" * 50)
    
    # Run all checks
    check_camera_detection()
    check_permissions()
    test_libcamera_commands()
    test_gstreamer()
    working_device, working_backend = test_opencv_backends()
    working_pipeline = test_specific_pipelines()
    
    # Summary
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if working_device and working_backend:
        print(f"✓ WORKING OPENCV METHOD: Device {working_device} with backend {working_backend}")
    else:
        print("✗ No working OpenCV method found")
    
    if working_pipeline:
        print(f"✓ WORKING GSTREAMER PIPELINE: {working_pipeline}")
    else:
        print("✗ No working GStreamer pipeline found")
    
    print("\nRECOMMENDATIONS:")
    
    # Check if user needs to logout/login
    success, stdout, stderr = run_command("groups")
    if success and "video" not in stdout:
        print("1. Add user to video group: sudo usermod -a -G video $USER")
        print("2. Log out and back in, or reboot")
    
    # Check if camera interface is enabled
    if not os.path.exists("/dev/video0"):
        print("3. Enable camera interface: sudo raspi-config -> Interface Options -> Camera")
        print("4. Reboot after enabling")
    
    # Check GStreamer plugins
    success, stdout, stderr = run_command("gst-inspect-1.0 libcamerasrc")
    if not success:
        print("5. Install GStreamer plugins: sudo apt install gstreamer1.0-libcamera")
    
    print("\nIf camera is detected by libcamera but OpenCV fails:")
    print("6. Try running as root: sudo python poc2.py")
    print("7. Check /boot/config.txt for camera settings")

if __name__ == "__main__":
    main()
