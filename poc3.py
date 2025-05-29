#!/usr/bin/env python3
"""
Live CCTV Detection Web Interface
Provides real-time camera feed with object detection overlay
Access via: http://localhost:8080
"""

import os
import cv2
import torch
import logging
import subprocess
import threading
import queue
import time
import json
import base64
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
import warnings
from flask import Flask, render_template, Response, jsonify, request
from collections import deque

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Config:
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 15
    YOLO_MODEL: str = 'yolov5n'
    CONFIDENCE_THRESHOLD: float = 0.4
    PROCESSING_RESOLUTION: Tuple[int, int] = (416, 416)
    WEB_PORT: int = 8080

# Global variables
camera = None
detector = None
current_frame = None
detection_stats = {"detections": [], "fps": 0, "frame_count": 0}
app = Flask(__name__)

class LibCameraCapture:
    def __init__(self, width=640, height=480, fps=15):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=3)
        self.process = None
        self.capture_thread = None
        self.running = False
        self.last_frame_time = time.time()
        
    def start(self) -> bool:
        """Start camera with fallback configurations"""
        configs = [
            {'codec': 'yuv420', 'width': self.width, 'height': self.height, 'fps': self.fps},
            {'codec': 'yuv420', 'width': 320, 'height': 240, 'fps': 10},
            {'codec': 'rgb', 'width': 320, 'height': 240, 'fps': 10}
        ]
        
        for i, config in enumerate(configs):
            logger.info(f"Trying camera config {i+1}: {config}")
            
            if self._start_with_config(config):
                logger.info(f"✓ Camera started with config {i+1}")
                return True
        
        logger.error("All camera configurations failed")
        return False
    
    def _start_with_config(self, config) -> bool:
        try:
            self._stop()
            
            cmd = [
                'libcamera-vid',
                '--width', str(config['width']),
                '--height', str(config['height']),
                '--framerate', str(config['fps']),
                '--timeout', '0',
                '--codec', config['codec'],
                '--output', '-',
                '--nopreview',
                '--flush'
            ]
            
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0, preexec_fn=os.setsid
            )
            
            time.sleep(2)
            
            if self.process.poll() is not None:
                return False
            
            # Update settings for this config
            self.width = config['width']
            self.height = config['height']
            self.fps = config['fps']
            self.codec = config['codec']
            
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            # Test frame capture
            start_time = time.time()
            while time.time() - start_time < 5:
                if not self.frame_queue.empty():
                    test_frame = self.frame_queue.get()
                    if test_frame is not None and test_frame.shape[0] > 0:
                        self.frame_queue.put(test_frame)  # Put it back
                        return True
                time.sleep(0.1)
            
            return False
            
        except Exception as e:
            logger.error(f"Config failed: {e}")
            return False
    
    def _capture_frames(self):
        """Capture frames based on codec"""
        if hasattr(self, 'codec') and self.codec == 'rgb':
            self._capture_rgb_frames()
        else:
            self._capture_yuv_frames()
    
    def _capture_rgb_frames(self):
        """Capture RGB frames"""
        frame_size = self.width * self.height * 3
        bytes_buffer = b""
        
        while self.running and self.process and self.process.poll() is None:
            try:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    time.sleep(0.001)
                    continue
                
                bytes_buffer += chunk
                
                while len(bytes_buffer) >= frame_size:
                    frame_data = bytes_buffer[:frame_size]
                    bytes_buffer = bytes_buffer[frame_size:]
                    
                    try:
                        rgb_data = np.frombuffer(frame_data, dtype=np.uint8)
                        rgb_frame = rgb_data.reshape((self.height, self.width, 3))
                        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                        
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self.frame_queue.put(bgr_frame, block=False)
                        self.last_frame_time = time.time()
                        
                    except Exception as e:
                        logger.error(f"RGB conversion error: {e}")
                        
            except Exception as e:
                logger.error(f"RGB capture error: {e}")
                time.sleep(0.001)
    
    def _capture_yuv_frames(self):
        """Capture YUV420 frames"""
        frame_size = self.width * self.height * 3 // 2
        bytes_buffer = b""
        
        while self.running and self.process and self.process.poll() is None:
            try:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    time.sleep(0.001)
                    continue
                
                bytes_buffer += chunk
                
                while len(bytes_buffer) >= frame_size:
                    frame_data = bytes_buffer[:frame_size]
                    bytes_buffer = bytes_buffer[frame_size:]
                    
                    try:
                        yuv_data = np.frombuffer(frame_data, dtype=np.uint8)
                        
                        # Try different YUV conversion methods
                        bgr_frame = None
                        for color_code in [cv2.COLOR_YUV2BGR_I420, cv2.COLOR_YUV2BGR_NV12]:
                            try:
                                yuv_frame = yuv_data.reshape((self.height * 3 // 2, self.width))
                                bgr_frame = cv2.cvtColor(yuv_frame, color_code)
                                if bgr_frame.shape == (self.height, self.width, 3):
                                    break
                            except Exception:
                                continue
                        
                        if bgr_frame is not None:
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            
                            self.frame_queue.put(bgr_frame, block=False)
                            self.last_frame_time = time.time()
                        
                    except Exception as e:
                        logger.error(f"YUV conversion error: {e}")
                        
            except Exception as e:
                logger.error(f"YUV capture error: {e}")
                time.sleep(0.001)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                return True, frame
            return False, None
        except Exception as e:
            logger.error(f"Frame read error: {e}")
            return False, None
    
    def _stop(self):
        """Stop capture"""
        self.running = False
        
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), 15)  # SIGTERM
                self.process.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.process.pid), 9)  # SIGKILL
                except Exception:
                    pass
            self.process = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def stop(self):
        """Stop camera"""
        self._stop()

class ThreatDetector:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.previous_frame = None
        self.motion_threshold = 1000
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            logger.info("Loading YOLO model...")
            self.model = torch.hub.load('ultralytics/yolov5', self.config.YOLO_MODEL, pretrained=True)
            self.model.conf = self.config.CONFIDENCE_THRESHOLD
            logger.info("✓ YOLO model loaded")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_objects(self, frame) -> Tuple[np.ndarray, List[dict], bool]:
        """Detect objects and return annotated frame + detection info"""
        if frame is None or frame.size == 0:
            return frame, [], False
        
        try:
            # Motion detection
            motion_detected, motion_score = self._detect_motion(frame)
            
            # Resize for YOLO
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                return frame, [], False
            
            new_w, new_h = self.config.PROCESSING_RESOLUTION
            frame_resized = cv2.resize(frame, (new_w, new_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # YOLO detection
            results = self.model(frame_rgb)
            detections = results.pandas().xyxy[0]
            
            # Scale detections back
            if not detections.empty:
                scale_x = w / new_w
                scale_y = h / new_h
                detections['xmin'] *= scale_x
                detections['xmax'] *= scale_x
                detections['ymin'] *= scale_y
                detections['ymax'] *= scale_y
            
            # Create annotated frame
            annotated_frame = frame.copy()
            detection_list = []
            threat_detected = False
            
            # Process detections
            for _, detection in detections.iterrows():
                if detection['confidence'] > self.config.CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
                    class_name = detection['name']
                    confidence = detection['confidence']
                    
                    # Validate coordinates
                    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
                    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
                    
                    # Determine threat level
                    is_threat = class_name in ['person', 'car', 'truck', 'bus']
                    if is_threat:
                        threat_detected = True
                    
                    # Color coding
                    if class_name == 'person':
                        color = (0, 255, 0)  # Green for person
                        threat_level = "HIGH" if motion_detected else "MEDIUM"
                    elif class_name in ['car', 'truck', 'bus']:
                        color = (0, 0, 255)  # Red for vehicles
                        threat_level = "HIGH" if motion_detected else "LOW"
                    else:
                        color = (255, 255, 0)  # Yellow for other objects
                        threat_level = "LOW"
                    
                    # Draw detection box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_name}: {confidence:.2f}"
                    if is_threat:
                        label += f" [{threat_level}]"
                    
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add to detection list
                    detection_list.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2],
                        'threat_level': threat_level if is_threat else 'LOW',
                        'is_threat': is_threat
                    })
            
            # Add motion indicator
            if motion_detected:
                cv2.putText(annotated_frame, f"MOTION: {motion_score:.0f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(annotated_frame, timestamp, (10, h-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return annotated_frame, detection_list, threat_detected
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return frame, [], False
    
    def _detect_motion(self, frame) -> Tuple[bool, float]:
        """Simple motion detection"""
        try:
            if self.previous_frame is None:
                self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return False, 0
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, self.previous_frame)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            motion_score = np.sum(thresh) / 255
            
            self.previous_frame = gray
            return motion_score > self.motion_threshold, motion_score
            
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return False, 0

def generate_frames():
    """Generate video frames for streaming"""
    global current_frame, detection_stats
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        if camera is None:
            time.sleep(0.1)
            continue
            
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Run object detection
        annotated_frame, detections, threat_detected = detector.detect_objects(frame)
        
        # Update stats
        current_time = time.time()
        fps = frame_count / (current_time - start_time) if current_time > start_time else 0
        
        detection_stats.update({
            'detections': detections,
            'fps': round(fps, 1),
            'frame_count': frame_count,
            'threat_detected': threat_detected,
            'timestamp': datetime.now().isoformat()
        })
        
        current_frame = annotated_frame
        
        # Encode frame as JPEG
        try:
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
        
        time.sleep(0.033)  # ~30 FPS max

# Flask routes
@app.route('/')
def index():
    """Main page"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Live CCTV Detection</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: white; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .video-container { display: flex; gap: 20px; margin-bottom: 20px; }
        .video-feed { flex: 2; }
        .stats-panel { flex: 1; background: #2a2a2a; padding: 15px; border-radius: 8px; }
        .video-feed img { width: 100%; height: auto; border: 2px solid #333; border-radius: 8px; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }
        .stat-item { background: #3a3a3a; padding: 10px; border-radius: 5px; }
        .stat-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .detections-list { max-height: 300px; overflow-y: auto; }
        .detection-item { background: #3a3a3a; margin: 5px 0; padding: 10px; border-radius: 5px; border-left: 4px solid #4CAF50; }
        .detection-item.threat { border-left-color: #f44336; }
        .controls { text-align: center; margin: 20px 0; }
        .btn { background: #4CAF50; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #45a049; }
        .threat-indicator { background: #f44336; color: white; padding: 10px; text-align: center; border-radius: 5px; margin-bottom: 15px; }
        .threat-indicator.safe { background: #4CAF50; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 Live CCTV Detection System</h1>
            <p>Real-time object detection with threat analysis</p>
        </div>
        
        <div id="threat-status" class="threat-indicator safe">
            System Status: Normal
        </div>
        
        <div class="video-container">
            <div class="video-feed">
                <img src="/video_feed" alt="Live Camera Feed">
            </div>
            
            <div class="stats-panel">
                <h3>📊 System Stats</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div>FPS</div>
                        <div class="stat-value" id="fps">0</div>
                    </div>
                    <div class="stat-item">
                        <div>Frames</div>
                        <div class="stat-value" id="frames">0</div>
                    </div>
                </div>
                
                <h3>🔍 Live Detections</h3>
                <div id="detections" class="detections-list">
                    <div style="text-align: center; color: #666;">No detections yet...</div>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="location.reload()">🔄 Refresh</button>
            <button class="btn" onclick="toggleFullscreen()">📺 Fullscreen</button>
            <button class="btn" onclick="downloadFrame()">📷 Capture</button>
        </div>
    </div>
    
    <script>
        // Auto-refresh stats every second
        setInterval(updateStats, 1000);
        
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps;
                    document.getElementById('frames').textContent = data.frame_count;
                    
                    // Update threat status
                    const threatStatus = document.getElementById('threat-status');
                    if (data.threat_detected) {
                        threatStatus.textContent = '⚠️ THREAT DETECTED!';
                        threatStatus.className = 'threat-indicator';
                    } else {
                        threatStatus.textContent = '✅ System Status: Normal';
                        threatStatus.className = 'threat-indicator safe';
                    }
                    
                    // Update detections
                    const detectionsDiv = document.getElementById('detections');
                    if (data.detections && data.detections.length > 0) {
                        detectionsDiv.innerHTML = data.detections.map(det => 
                            `<div class="detection-item ${det.is_threat ? 'threat' : ''}">
                                <strong>${det.class}</strong> (${(det.confidence * 100).toFixed(1)}%)
                                <br><small>Threat Level: ${det.threat_level}</small>
                            </div>`
                        ).join('');
                    } else {
                        detectionsDiv.innerHTML = '<div style="text-align: center; color: #666;">No detections</div>';
                    }
                })
                .catch(err => console.error('Stats update failed:', err));
        }
        
        function toggleFullscreen() {
            const video = document.querySelector('.video-feed img');
            if (video.requestFullscreen) {
                video.requestFullscreen();
            }
        }
        
        function downloadFrame() {
            const link = document.createElement('a');
            link.href = '/capture';
            link.download = 'capture_' + new Date().toISOString().slice(0,19).replace(/:/g, '-') + '.jpg';
            link.click();
        }
    </script>
</body>
</html>
    '''

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Get current detection stats"""
    return jsonify(detection_stats)

@app.route('/capture')
def capture():
    """Capture current frame"""
    global current_frame
    if current_frame is not None:
        ret, buffer = cv2.imencode('.jpg', current_frame)
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    return "No frame available", 404

def initialize_system():
    """Initialize camera and detector"""
    global camera, detector
    
    try:
        # Initialize camera
        logger.info("Initializing camera...")
        config = Config()
        camera = LibCameraCapture(config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS)
        
        if not camera.start():
            logger.error("Failed to start camera")
            return False
        
        # Initialize detector
        logger.info("Initializing object detector...")
        detector = ThreatDetector(config)
        
        logger.info("✅ System initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting Live CCTV Detection Web Interface")
    
    if not initialize_system():
        logger.error("❌ System initialization failed")
        return False
    
    try:
        logger.info(f"🌐 Starting web server on http://localhost:{Config.WEB_PORT}")
        logger.info("Press Ctrl+C to stop")
        
        app.run(
            host='0.0.0.0',
            port=Config.WEB_PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.info("Stopping web interface...")
    except Exception as e:
        logger.error(f"Web server error: {e}")
    finally:
        if camera:
            camera.stop()
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()
