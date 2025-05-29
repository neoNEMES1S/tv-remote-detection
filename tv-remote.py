#!/usr/bin/env python3
"""
TV Remote Pickup Detection System - Fixed Version
Specifically designed to detect when someone picks up a TV remote in a living room
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
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from flask import Flask, render_template_string, Response, jsonify
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Config:
    # Camera settings
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 15
    
    # AI Model settings
    YOLO_MODEL: str = 'yolov5n'
    CONFIDENCE_THRESHOLD: float = 0.3
    PROCESSING_RESOLUTION: Tuple[int, int] = (416, 416)
    
    # Remote detection settings
    REMOTE_KEYWORDS: List[str] = ['remote', 'cell phone', 'remote control']
    PERSON_REMOTE_DISTANCE_THRESHOLD: int = 100
    
    # Motion sensitivity
    MOTION_THRESHOLD: int = 800
    PICKUP_MOTION_THRESHOLD: int = 1200
    
    # Recording settings
    CLIP_DURATION_SECONDS: int = 8
    OUTPUT_DIR: str = "remote_pickups"
    
    # Web interface
    WEB_PORT: int = 8080

# Global variables
camera = None
detector = None
current_frame = None
detection_stats = {
    "remote_pickups": 0,
    "person_detected": False,
    "remote_detected": False,
    "pickup_detected": False,
    "fps": 0,
    "timestamp": ""
}

app = Flask(__name__)

class LibCameraCapture:
    """Simple camera capture for Pi Camera"""
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
        """Start camera with simple libcamera-vid"""
        try:
            self._stop()
            
            cmd = [
                'libcamera-vid',
                '--width', str(self.width),
                '--height', str(self.height),
                '--framerate', str(self.fps),
                '--timeout', '0',
                '--codec', 'yuv420',
                '--output', '-',
                '--nopreview',
                '--flush'
            ]
            
            logger.info("Starting camera...")
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0, preexec_fn=os.setsid
            )
            
            time.sleep(2)
            
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                logger.error(f"Camera failed: {stderr_output}")
                return False
            
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            # Test frame capture
            start_time = time.time()
            while time.time() - start_time < 5:
                if not self.frame_queue.empty():
                    test_frame = self.frame_queue.get()
                    if test_frame is not None and test_frame.shape[0] > 0:
                        logger.info(f"✓ Camera started: {test_frame.shape}")
                        self.frame_queue.put(test_frame)
                        return True
                time.sleep(0.1)
            
            logger.error("No frames received")
            return False
            
        except Exception as e:
            logger.error(f"Camera start failed: {e}")
            return False
    
    def _capture_frames(self):
        """Capture YUV420 frames"""
        frame_size = self.width * self.height * 3 // 2
        bytes_buffer = b""
        
        logger.info("Frame capture thread started")
        
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
                        yuv_frame = yuv_data.reshape((self.height * 3 // 2, self.width))
                        bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                        
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self.frame_queue.put(bgr_frame, block=False)
                        self.last_frame_time = time.time()
                        
                    except Exception as e:
                        logger.error(f"Frame conversion error: {e}")
                        
            except Exception as e:
                time.sleep(0.001)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                return True, frame
            return False, None
        except Exception:
            return False, None
    
    def _stop(self):
        self.running = False
        
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), 15)
                self.process.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.process.pid), 9)
                except Exception:
                    pass
            self.process = None
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def stop(self):
        self._stop()

class RemoteDetector:
    """TV Remote pickup detector"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.previous_frame = None
        self.frame_buffer = deque(maxlen=config.CLIP_DURATION_SECONDS * config.CAMERA_FPS)
        self._load_model()
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
    def _load_model(self):
        try:
            logger.info("Loading YOLO model...")
            self.model = torch.hub.load('ultralytics/yolov5', self.config.YOLO_MODEL, pretrained=True)
            self.model.conf = self.config.CONFIDENCE_THRESHOLD
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def detect_remote_pickup(self, frame) -> Tuple[np.ndarray, Dict]:
        """Main detection function"""
        if frame is None or frame.size == 0:
            return frame, self._empty_result()
        
        try:
            self.frame_buffer.append(frame.copy())
            
            # Motion detection
            motion_detected, motion_score = self._detect_motion(frame)
            
            # YOLO detection
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                return frame, self._empty_result()
            
            # Resize for YOLO
            new_w, new_h = self.config.PROCESSING_RESOLUTION
            frame_resized = cv2.resize(frame, (new_w, new_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Run detection
            results = self.model(frame_rgb)
            detections = results.pandas().xyxy[0]
            
            # Scale back to original size
            if not detections.empty:
                scale_x = w / new_w
                scale_y = h / new_h
                detections['xmin'] *= scale_x
                detections['xmax'] *= scale_x
                detections['ymin'] *= scale_y
                detections['ymax'] *= scale_y
            
            # Analyze for pickup
            result = self._analyze_remote_pickup(frame, detections, motion_detected, motion_score)
            
            # Create annotated frame
            annotated_frame = self._draw_detections(frame, detections, result)
            
            return annotated_frame, result
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return frame, self._empty_result()
    
    def _analyze_remote_pickup(self, frame, detections, motion_detected, motion_score) -> Dict:
        """Analyze if pickup is occurring"""
        
        result = {
            'person_detected': False,
            'remote_detected': False,
            'pickup_detected': False,
            'pickup_confidence': 0.0,
            'motion_score': motion_score,
            'distance_to_remote': None,
            'pickup_reason': '',
            'timestamp': datetime.now().isoformat()
        }
        
        if detections.empty:
            return result
        
        # Find person
        persons = detections[detections['name'] == 'person']
        if not persons.empty:
            result['person_detected'] = True
            person = persons.loc[persons['confidence'].idxmax()]
            person_bbox = [int(person['xmin']), int(person['ymin']), 
                          int(person['xmax']), int(person['ymax'])]
        
        # Find remote
        remote_objects = detections[detections['name'].isin(self.config.REMOTE_KEYWORDS)]
        if not remote_objects.empty:
            result['remote_detected'] = True
            remote = remote_objects.loc[remote_objects['confidence'].idxmax()]
            remote_bbox = [int(remote['xmin']), int(remote['ymin']), 
                          int(remote['xmax']), int(remote['ymax'])]
        
        # Check for pickup
        if result['person_detected'] and result['remote_detected']:
            person_center = self._get_bbox_center(person_bbox)
            remote_center = self._get_bbox_center(remote_bbox)
            distance = self._calculate_distance(person_center, remote_center)
            result['distance_to_remote'] = distance
            
            # Simple pickup logic
            confidence = 0.0
            reasons = []
            
            if distance < self.config.PERSON_REMOTE_DISTANCE_THRESHOLD:
                confidence += 0.4
                reasons.append(f"Close proximity ({distance:.0f}px)")
            
            if motion_detected:
                confidence += 0.3 if motion_score > self.config.PICKUP_MOTION_THRESHOLD else 0.15
                reasons.append(f"Motion detected ({motion_score:.0f})")
            
            if remote_center[1] > person_bbox[1] + (person_bbox[3] - person_bbox[1]) * 0.3:
                confidence += 0.2
                reasons.append("Remote in hand area")
            
            result['pickup_confidence'] = min(confidence, 1.0)
            result['pickup_reason'] = "; ".join(reasons) if reasons else "No pickup indicators"
            
            if confidence > 0.6:
                result['pickup_detected'] = True
                self._record_pickup_event(result)
        
        return result
    
    def _record_pickup_event(self, result):
        """Record pickup event"""
        try:
            if len(self.frame_buffer) < 10:
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"remote_pickup_{timestamp}.mp4"
            output_path = os.path.join(self.config.OUTPUT_DIR, filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.config.CAMERA_FPS, 
                                (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
            
            frames_written = 0
            for frame in self.frame_buffer:
                if frame is not None:
                    out.write(frame)
                    frames_written += 1
            
            out.release()
            
            global detection_stats
            detection_stats['remote_pickups'] += 1
            
            logger.info(f"📹 Recorded pickup: {filename} ({frames_written} frames)")
            
        except Exception as e:
            logger.error(f"Recording failed: {e}")
    
    def _detect_motion(self, frame) -> Tuple[bool, float]:
        """Motion detection"""
        try:
            if self.previous_frame is None:
                self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return False, 0
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, self.previous_frame)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_score = np.sum(thresh) / 255
            
            self.previous_frame = gray
            return motion_score > self.config.MOTION_THRESHOLD, motion_score
            
        except Exception as e:
            return False, 0
    
    def _draw_detections(self, frame, detections, result) -> np.ndarray:
        """Draw detection boxes"""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        try:
            # Draw detections
            for _, detection in detections.iterrows():
                if detection['confidence'] > self.config.CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
                    name = detection['name']
                    conf = detection['confidence']
                    
                    # Color based on object type
                    if name == 'person':
                        color = (0, 255, 0)  # Green
                    elif name in self.config.REMOTE_KEYWORDS:
                        color = (255, 0, 0)  # Blue
                    else:
                        color = (0, 255, 255)  # Yellow
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{name}: {conf:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Status overlay
            if result['pickup_detected']:
                cv2.putText(annotated_frame, "PICKUP DETECTED!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif result['person_detected'] and result['remote_detected']:
                cv2.putText(annotated_frame, "MONITORING...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Motion indicator
            if result['motion_score'] > self.config.MOTION_THRESHOLD:
                cv2.putText(annotated_frame, f"MOTION: {result['motion_score']:.0f}", 
                           (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Timestamp
            timestamp = datetime.now().strftime('%H:%M:%S')
            cv2.putText(annotated_frame, timestamp, (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            logger.error(f"Drawing error: {e}")
        
        return annotated_frame
    
    def _get_bbox_center(self, bbox) -> Tuple[int, int]:
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
    def _calculate_distance(self, point1, point2) -> float:
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _empty_result(self) -> Dict:
        return {
            'person_detected': False,
            'remote_detected': False,
            'pickup_detected': False,
            'pickup_confidence': 0.0,
            'motion_score': 0.0,
            'distance_to_remote': None,
            'pickup_reason': '',
            'timestamp': datetime.now().isoformat()
        }

def generate_frames():
    """Generate frames for web streaming"""
    global current_frame, detection_stats
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        if camera is None or detector is None:
            time.sleep(0.1)
            continue
            
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Run detection
        annotated_frame, result = detector.detect_remote_pickup(frame)
        
        # Update stats
        current_time = time.time()
        fps = frame_count / (current_time - start_time) if current_time > start_time else 0
        
        detection_stats.update({
            'person_detected': result['person_detected'],
            'remote_detected': result['remote_detected'],
            'pickup_detected': result['pickup_detected'],
            'pickup_confidence': result['pickup_confidence'],
            'motion_score': result['motion_score'],
            'fps': round(fps, 1),
            'frame_count': frame_count,
            'timestamp': result['timestamp']
        })
        
        current_frame = annotated_frame
        
        # Encode frame
        try:
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
        
        time.sleep(0.033)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>TV Remote Pickup Detection</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #1a1a1a; 
            color: white; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        .header { 
            text-align: center; 
            margin-bottom: 20px; 
        }
        .main-content { 
            display: flex; 
            gap: 20px; 
            margin-bottom: 20px; 
        }
        .video-section { flex: 2; }
        .stats-section { 
            flex: 1; 
            background-color: #2a2a2a; 
            padding: 20px; 
            border-radius: 10px; 
        }
        .video-feed { 
            width: 100%; 
            height: auto; 
            border: 3px solid #333; 
            border-radius: 10px; 
        }
        .pickup-alert { 
            background-color: #f44336; 
            color: white; 
            padding: 15px; 
            text-align: center; 
            border-radius: 8px; 
            margin-bottom: 15px; 
            font-size: 18px; 
            font-weight: bold; 
        }
        .pickup-alert.monitoring { background-color: #ff9800; }
        .pickup-alert.safe { background-color: #4CAF50; }
        .stats-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 15px; 
            margin-bottom: 20px; 
        }
        .stat-card { 
            background-color: #3a3a3a; 
            padding: 15px; 
            border-radius: 8px; 
            text-align: center; 
        }
        .stat-value { 
            font-size: 28px; 
            font-weight: bold; 
            color: #4CAF50; 
            margin-bottom: 5px; 
        }
        .detection-details { 
            background-color: #3a3a3a; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 15px; 
        }
        .controls { 
            text-align: center; 
            margin: 20px 0; 
        }
        .btn { 
            background-color: #4CAF50; 
            color: white; 
            border: none; 
            padding: 12px 24px; 
            margin: 5px; 
            border-radius: 6px; 
            cursor: pointer; 
            font-size: 16px; 
        }
        .btn:hover { background-color: #45a049; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📺 TV Remote Pickup Detection</h1>
            <p>Living Room Monitoring System</p>
        </div>
        
        <div id="pickup-status" class="pickup-alert safe">
            🟢 System Active - No pickup detected
        </div>
        
        <div class="main-content">
            <div class="video-section">
                <img src="/video_feed" alt="Live Camera Feed" class="video-feed">
            </div>
            
            <div class="stats-section">
                <h3>📊 Detection Status</h3>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="total-pickups">0</div>
                        <div>Total Pickups</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="fps">0</div>
                        <div>FPS</div>
                    </div>
                </div>
                
                <div class="detection-details">
                    <h4>🎯 Current Status</h4>
                    <p>Person: <span id="person-detected">No</span></p>
                    <p>Remote: <span id="remote-detected">No</span></p>
                    <p>Confidence: <span id="confidence">0%</span></p>
                    <p>Motion: <span id="motion">0</span></p>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="location.reload()">🔄 Refresh</button>
            <button class="btn" onclick="captureFrame()">📷 Capture</button>
        </div>
    </div>
    
    <script>
        setInterval(updateStats, 500);
        
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-pickups').textContent = data.remote_pickups || 0;
                    document.getElementById('fps').textContent = data.fps || 0;
                    document.getElementById('person-detected').textContent = data.person_detected ? 'Yes' : 'No';
                    document.getElementById('remote-detected').textContent = data.remote_detected ? 'Yes' : 'No';
                    document.getElementById('confidence').textContent = Math.round((data.pickup_confidence || 0) * 100) + '%';
                    document.getElementById('motion').textContent = Math.round(data.motion_score || 0);
                    
                    const statusDiv = document.getElementById('pickup-status');
                    if (data.pickup_detected) {
                        statusDiv.textContent = '🚨 REMOTE PICKUP DETECTED!';
                        statusDiv.className = 'pickup-alert';
                    } else if (data.person_detected && data.remote_detected) {
                        statusDiv.textContent = '👀 Person and Remote Detected - Monitoring...';
                        statusDiv.className = 'pickup-alert monitoring';
                    } else {
                        statusDiv.textContent = '🟢 System Active - No pickup detected';
                        statusDiv.className = 'pickup-alert safe';
                    }
                })
                .catch(err => console.error('Stats update failed:', err));
        }
        
        function captureFrame() {
            const link = document.createElement('a');
            link.href = '/capture';
            link.download = 'remote_detection_' + Date.now() + '.jpg';
            link.click();
        }
        
        updateStats();
    </script>
</body>
</html>
"""

# Flask Routes
@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(detection_stats)

@app.route('/capture')
def capture():
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
        logger.info("🚀 Initializing TV Remote Detection System...")
        
        config = Config()
        camera = LibCameraCapture(config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS)
        
        if not camera.start():
            logger.error("❌ Failed to start camera")
            return False
        
        detector = RemoteDetector(config)
        
        logger.info("✅ System initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False

def main():
    logger.info("📺 Starting TV Remote Pickup Detection System")
    
    if not initialize_system():
        logger.error("❌ System initialization failed")
        return False
    
    try:
        logger.info(f"🌐 Web interface: http://localhost:{Config.WEB_PORT}")
        logger.info("Press Ctrl+C to stop")
        
        app.run(
            host='0.0.0.0',
            port=Config.WEB_PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Stopping...")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        if camera:
            camera.stop()
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    main()
