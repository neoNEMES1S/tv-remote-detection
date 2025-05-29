# 📺 TV Remote Pickup Detection System

An intelligent computer vision system designed to detect when someone picks up a TV remote control. Built specifically for Raspberry Pi 5 with Camera Module 3, featuring multiple detection strategies and low-light optimization.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-red.svg)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-ff69b4.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Overview

This project implements a real-time anomaly detection system that monitors a room and detects when someone picks up a TV remote control. It uses advanced computer vision techniques including:

- **YOLOv11/v5** for object detection
- **Multi-strategy detection** (color, contour, edge detection)
- **Motion analysis** for pickup event detection
- **Low-light optimization** for 24/7 monitoring
- **Web-based monitoring interface**

## ✨ Features

- **Real-time Detection**: Processes video at 5-10 FPS on Raspberry Pi 5
- **Multiple Detection Strategies**:
  - YOLO-based object detection
  - Color-based detection (for black remotes)
  - Contour detection (for rectangular shapes)
  - Edge detection (for remote outlines)
- **Smart Pickup Detection**: Analyzes person-remote proximity and motion
- **Low-Light Optimization**: CLAHE enhancement and optimized camera settings
- **Event Recording**: Automatically saves video clips of pickup events
- **Web Interface**: Real-time monitoring dashboard
- **Performance Optimized**: Frame skipping and efficient processing

## 📋 Requirements

### Hardware
- Raspberry Pi 5 (4GB or 8GB recommended)
- Raspberry Pi Camera Module 3
- MicroSD card (32GB+ recommended)
- Adequate lighting or IR illumination for night vision

### Software
- Raspberry Pi OS (64-bit recommended)
- Python 3.9+
- OpenCV 4.8.1
- Flask 3.0.0
- PyTorch 2.0.0 (optional, for YOLO support)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/tv-remote-detection.git
cd tv-remote-detection
```

### 2. Install Dependencies

#### Option A: Automated Installation
```bash
chmod +x install.sh
./install.sh
```

#### Option B: Manual Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option C: Lightweight Installation (No PyTorch)
```bash
pip install -r requirements-lightweight.txt
```

### 3. Run the System
```bash
# Activate virtual environment
source venv/bin/activate

# Run the detection system
python tv-remote-heavy.py
```

### 4. Access Web Interface
Open your browser and navigate to:
```
http://raspberrypi.local:8080
```
or
```
http://<raspberry-pi-ip>:8080
```

## 🔧 Configuration

Edit the `Config` class in `tv-remote-heavy.py` to customize:

```python
class Config:
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 10
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.15
    DETECTION_SKIP_FRAMES = 2
    
    # Remote detection
    ENABLE_COLOR_DETECTION = True
    ENABLE_CONTOUR_DETECTION = True
    ENABLE_EDGE_DETECTION = True
```

### Adjusting for Your Remote

If your remote has specific colors, modify:
```python
# For dark gray remotes
BLACK_LOWER_HSV = [0, 0, 0]
BLACK_UPPER_HSV = [180, 50, 80]
```

## 📊 Performance Optimization

### Current Performance
- **Raspberry Pi 5 (4GB)**: 5-8 FPS with full detection
- **Inference Time**: ~100-200ms per frame
- **Memory Usage**: ~500MB

### Optimization Tips
1. **Increase frame skipping**: Set `DETECTION_SKIP_FRAMES = 3`
2. **Reduce resolution**: Lower `PROCESSING_RESOLUTION = (256, 256)`
3. **Disable strategies**: Turn off unused detection methods
4. **Use lightweight mode**: Run without PyTorch/YOLO

## 🎯 Improving Detection Accuracy

### 1. Lighting
- Add LED lighting near the monitoring area
- Avoid backlighting
- Use IR illumination for night monitoring

### 2. Camera Position
- Mount camera 4-6 feet high
- Angle to see remotes clearly
- Ensure remote is within 2-3 feet of camera

### 3. Remote Marking
- Add a small bright sticker to your remote
- Use a remote with contrasting colors

### 4. Fine-tuning
- Adjust confidence thresholds
- Modify size ranges for your specific remote
- Train a custom YOLO model on your remote

## 📁 Project Structure

```
tv-remote-detection/
├── tv-remote-heavy.py      # Main detection script
├── requirements.txt        # Full dependencies
├── requirements-lightweight.txt  # Minimal dependencies
├── install.sh             # Installation script
├── README.md              # This file
└── remote_pickups/        # Recorded pickup events
    ├── remote_pickup_YYYYMMDD_HHMMSS.mp4
    └── remote_pickup_YYYYMMDD_HHMMSS.json
```

## 🖥️ Web Interface

The web interface provides:
- Live camera feed with detection overlays
- Real-time statistics (FPS, detections, confidence)
- Detection method indicators
- Performance metrics
- Event history

### Interface Features
- **Status Indicators**: Shows current detection state
- **Confidence Bar**: Visual representation of pickup likelihood
- **Method Tags**: Shows which detection strategies found the remote
- **Performance Stats**: FPS, inference time, frame count

## 🐛 Troubleshooting

### Common Issues

1. **Low FPS (< 3 FPS)**
   - Increase `DETECTION_SKIP_FRAMES`
   - Reduce `PROCESSING_RESOLUTION`
   - Disable unused detection strategies

2. **Remote Not Detected**
   - Ensure good lighting
   - Adjust `CONFIDENCE_THRESHOLD` lower
   - Try different detection strategies
   - Check remote size is within `REMOTE_SIZE_RANGE`

3. **PyTorch Installation Fails**
   - Use lightweight version without PyTorch
   - Try system packages: `sudo apt install python3-torch`
   - Download ARM-specific wheels

4. **Camera Not Starting**
   - Check camera connection
   - Run with sudo: `sudo python tv-remote-heavy.py`
   - Verify camera with: `libcamera-hello`

### Debug Mode
Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv5/v11 by Ultralytics
- OpenCV community
- Flask web framework
- Raspberry Pi Foundation

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section

## 🚀 Future Enhancements

- [ ] Custom YOLO model training for TV remotes
- [ ] Multi-camera support
- [ ] Cloud storage integration
- [ ] Mobile app for notifications
- [ ] Advanced analytics dashboard
- [ ] Integration with home automation systems
- [ ] Support for multiple remote types

---

**Note**: This system is designed for personal/educational use. Ensure you comply with local privacy laws when deploying surveillance systems.
