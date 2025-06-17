# Enhanced HTML Template - Clean Minimalistic Design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Security Detection System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
        }
        
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 300;
            margin-bottom: 10px;
            color: white;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.8;
            font-weight: 300;
        }
        
        .alert-banner {
            background: rgba(76, 175, 80, 0.9);
            padding: 15px 25px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 25px;
            font-size: 1.1rem;
            font-weight: 500;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .alert-banner.hands-up { 
            background: rgba(244, 67, 54, 0.9); 
            animation: pulse 1s infinite;
        }
        
        .alert-banner.monitoring { 
            background: rgba(255, 152, 0, 0.9); 
        }
        
        .alert-banner.pickup { 
            background: rgba(33, 150, 243, 0.9); 
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        .main-content { 
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 25px;
            margin-bottom: 25px;
        }
        
        .video-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .video-feed { 
            width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .stats-section { 
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .stats-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stats-card h3 {
            font-size: 1.3rem;
            margin-bottom: 20px;
            font-weight: 500;
            color: white;
        }
        
        .stats-grid { 
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-item { 
            text-align: center;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stat-value { 
            font-size: 2rem;
            font-weight: 600;
            color: #4CAF50;
            margin-bottom: 5px;
            display: block;
        }
        
        .stat-value.alert { color: #f44336; }
        .stat-value.warning { color: #ff9800; }
        
        .stat-label {
            font-size: 0.9rem;
            opacity: 0.8;
            font-weight: 400;
        }
        
        .detection-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .detection-row:last-child {
            border-bottom: none;
        }
        
        .detection-label {
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4CAF50;
            transition: all 0.3s ease;
        }
        
        .status-dot.active { background: #f44336; }
        .status-dot.warning { background: #ff9800; }
        
        .detection-value {
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.1);
            font-size: 0.9rem;
        }
        
        .controls { 
            text-align: center;
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .btn { 
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .btn:hover { 
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        
        .btn.emergency { 
            background: rgba(244, 67, 54, 0.8);
            border-color: rgba(244, 67, 54, 0.6);
        }
        
        .btn.emergency:hover { 
            background: rgba(244, 67, 54, 1);
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Security Detection System</h1>
            <p>Live monitoring with AI detection</p>
        </div>
        
        <div id="alert-banner" class="alert-banner">
            🟢 System Active - All Clear
        </div>
        
        <div class="main-content">
            <div class="video-section">
                <img src="/video_feed" alt="Live Feed" class="video-feed">
            </div>
            
            <div class="stats-section">
                <div class="stats-card">
                    <h3>📊 System Status</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-value" id="fps">0</span>
                            <div class="stat-label">FPS</div>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value alert" id="total-alerts">0</span>
                            <div class="stat-label">Total Alerts</div>
                        </div>
                    </div>
                </div>
                
                <div class="stats-card">
                    <h3>🎯 Detection Status</h3>
                    <div class="detection-row">
                        <div class="detection-label">
                            <div class="status-dot" id="person-dot"></div>
                            Person
                        </div>
                        <div class="detection-value" id="person-status">No</div>
                    </div>
                    <div class="detection-row">
                        <div class="detection-label">
                            <div class="status-dot" id="remote-dot"></div>
                            Remote
                        </div>
                        <div class="detection-value" id="remote-status">No</div>
                    </div>
                    <div class="detection-row">
                        <div class="detection-label">
                            <div class="status-dot" id="pickup-dot"></div>
                            Pickup
                        </div>
                        <div class="detection-value" id="pickup-status">0%</div>
                    </div>
                </div>
                
                <div class="stats-card">
                    <h3>🚨 Security Monitor</h3>
                    <div class="detection-row">
                        <div class="detection-label">
                            <div class="status-dot" id="pose-dot"></div>
                            Pose
                        </div>
                        <div class="detection-value" id="pose-status">No</div>
                    </div>
                    <div class="detection-row">
                        <div class="detection-label">
                            <div class="status-dot" id="hands-dot"></div>
                            Hands Up
                        </div>
                        <div class="detection-value" id="hands-status">0%</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="location.reload()">🔄 Refresh</button>
            <button class="btn" onclick="captureFrame()">📷 Capture</button>
            <button class="btn emergency" onclick="emergencyAlert()">🚨 Emergency</button>
        </div>
    </div>
    
    <script>
        let alertSound = null;
        let lastHandsUpAlert = false;
        
        function initAudio() {
            try {
                alertSound = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigE=');
            } catch (e) {
                console.log('Audio initialization failed');
            }
        }
        
        setInterval(updateStats, 500);
        
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    // Update basic stats
                    document.getElementById('fps').textContent = data.fps || 0;
                    document.getElementById('total-alerts').textContent = (data.remote_pickups || 0) + (data.hands_up_alerts || 0);
                    
                    // Update detection status with dots
                    updateStatusDot('person-dot', data.person_detected);
                    updateStatusDot('remote-dot', data.remote_detected);
                    updateStatusDot('pickup-dot', data.pickup_detected);
                    updateStatusDot('pose-dot', data.pose_detected);
                    updateStatusDot('hands-dot', data.hands_up_detected);
                    
                    document.getElementById('person-status').textContent = data.person_detected ? 'Yes' : 'No';
                    document.getElementById('remote-status').textContent = data.remote_detected ? 'Yes' : 'No';
                    document.getElementById('pickup-status').textContent = Math.round((data.pickup_confidence || 0) * 100) + '%';
                    document.getElementById('pose-status').textContent = data.pose_detected ? 'Yes' : 'No';
                    document.getElementById('hands-status').textContent = Math.round((data.hands_up_confidence || 0) * 100) + '%';
                    
                    // Update main alert banner
                    const alertBanner = document.getElementById('alert-banner');
                    
                    if (data.hands_up_detected) {
                        alertBanner.textContent = '🚨 HANDS UP DETECTED - EMERGENCY ALERT!';
                        alertBanner.className = 'alert-banner hands-up';
                        
                        if (!lastHandsUpAlert && alertSound) {
                            alertSound.play().catch(e => console.log('Audio play failed'));
                        }
                        lastHandsUpAlert = true;
                    } else if (data.pickup_detected) {
                        alertBanner.textContent = '📺 Remote Detected in Hand';
                        alertBanner.className = 'alert-banner pickup';
                        lastHandsUpAlert = false;
                    } else if (data.person_detected && data.remote_detected) {
                        alertBanner.textContent = '👀 Person and Remote Detected';
                        alertBanner.className = 'alert-banner monitoring';
                        lastHandsUpAlert = false;
                    } else if (data.hands_up_confidence > 0.3) {
                        alertBanner.textContent = '⚠️ Monitoring Hand Position...';
                        alertBanner.className = 'alert-banner monitoring';
                        lastHandsUpAlert = false;
                    } else {
                        alertBanner.textContent = '🟢 System Active - All Clear';
                        alertBanner.className = 'alert-banner';
                        lastHandsUpAlert = false;
                    }
                })
                .catch(err => console.error('Stats update failed:', err));
        }
        
        function updateStatusDot(dotId, isActive) {
            const dot = document.getElementById(dotId);
            if (isActive) {
                dot.className = 'status-dot active';
            } else {
                dot.className = 'status-dot';
            }
        }
        
        function captureFrame() {
            const link = document.createElement('a');
            link.href = '/capture';
            link.download = 'security_capture_' + Date.now() + '.jpg';
            link.click();
        }
        
        function emergencyAlert() {
            if (confirm('Send emergency alert?')) {
                fetch('/emergency_alert', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        alert('Emergency alert sent! ' + data.message);
                    })
                    .catch(err => {
                        alert('Emergency alert failed!');
                        console.error(err);
                    });
            }
        }
        
        initAudio();
        updateStats();
    </script>
</body>
</html>
"""#!/usr/bin/env python3
"""
Enhanced Detection System - TV Remote + Hands-Up Detection
Detects TV remote pickups and potential hostage situations (hands up)
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
from typing import List, Optional, Tuple, Dict, Union, Any
from flask import Flask, render_template_string, Response, jsonify
from collections import deque
import warnings
import mediapipe as mp

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
    CONFIDENCE_THRESHOLD: float = 0.25  # Lowered for better remote detection
    PROCESSING_RESOLUTION: Tuple[int, int] = (416, 416)
    
    # Remote detection settings - simplified
    REMOTE_KEYWORDS: List[str] = ['remote', 'cell phone', 'remote control']
    PERSON_REMOTE_DISTANCE_THRESHOLD: int = 150  # Increased for easier detection
    
    # Hands-up detection settings
    HANDS_UP_THRESHOLD: float = 0.7
    HANDS_UP_DURATION_THRESHOLD: int = 3
    WRIST_SHOULDER_RATIO: float = -0.1
    
    # Motion sensitivity
    MOTION_THRESHOLD: int = 800
    PICKUP_MOTION_THRESHOLD: int = 1200
    
    # Recording settings
    CLIP_DURATION_SECONDS: int = 8
    OUTPUT_DIR: str = "detection_events"
    
    # Web interface
    WEB_PORT: int = 8080

# Global variables
camera = None
detector = None
current_frame = None
detection_stats = {
    "remote_pickups": 0,
    "hands_up_alerts": 0,
    "person_detected": False,
    "remote_detected": False,
    "pickup_detected": False,
    "hands_up_detected": False,
    "hands_up_confidence": 0.0,
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
        """Fixed type hint for read method"""
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

class EnhancedDetector:
    """Enhanced detector for TV Remote pickup and hands-up situations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.pose_detector = None
        self.previous_frame = None
        self.hands_up_start_time = None
        self.hands_up_confirmed = False
        self._load_models()
        
    def _load_models(self):
        try:
            logger.info("Loading YOLO model...")
            self.model = torch.hub.load('ultralytics/yolov5', self.config.YOLO_MODEL, pretrained=True)
            self.model.conf = self.config.CONFIDENCE_THRESHOLD
            logger.info("✓ YOLO model loaded successfully")
            
            logger.info("Loading MediaPipe Pose model...")
            mp_pose = mp.solutions.pose
            self.pose_detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("✓ MediaPipe Pose model loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def detect_events(self, frame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Main detection function for both remote pickup and hands-up"""
        if frame is None or frame.size == 0:
            return frame, self._empty_result()
        
        try:
            # Motion detection
            motion_detected, motion_score = self._detect_motion(frame)
            
            # YOLO detection for objects
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                return frame, self._empty_result()
            
            # Resize for YOLO
            new_w, new_h = self.config.PROCESSING_RESOLUTION
            frame_resized = cv2.resize(frame, (new_w, new_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection
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
            
            # MediaPipe pose detection for hands-up
            pose_results = self._detect_pose(frame)
            
            # Analyze for remote pickup
            pickup_result = self._analyze_remote_pickup(frame, detections, motion_detected, motion_score)
            
            # Analyze for hands-up situation
            hands_up_result = self._analyze_hands_up(frame, pose_results)
            
            # Combine results
            result = {**pickup_result, **hands_up_result}
            
            # Create annotated frame
            annotated_frame = self._draw_all_detections(frame, detections, pose_results, result)
            
            return annotated_frame, result
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return frame, self._empty_result()
    
    def _detect_pose(self, frame):
        """Detect human pose using MediaPipe"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(frame_rgb)
            return results
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
            return None
    
    def _analyze_hands_up(self, frame, pose_results) -> Dict[str, Any]:
        """Analyze if person has hands up (potential hostage situation)"""
        result = {
            'hands_up_detected': False,
            'hands_up_confidence': 0.0,
            'hands_up_duration': 0.0,
            'pose_detected': False,
            'left_hand_up': False,
            'right_hand_up': False,
            'hands_up_reason': ''
        }
        
        if not pose_results or not pose_results.pose_landmarks:
            self.hands_up_start_time = None
            self.hands_up_confirmed = False
            return result
        
        result['pose_detected'] = True
        landmarks = pose_results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        
        try:
            # Get key landmarks
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            
            # Check if hands are up (wrists above shoulders)
            left_hand_up = (left_wrist.y < left_shoulder.y + self.config.WRIST_SHOULDER_RATIO and 
                           left_wrist.visibility > 0.5)
            right_hand_up = (right_wrist.y < right_shoulder.y + self.config.WRIST_SHOULDER_RATIO and 
                            right_wrist.visibility > 0.5)
            
            result['left_hand_up'] = left_hand_up
            result['right_hand_up'] = right_hand_up
            
            # Calculate confidence based on hand positions
            confidence = 0.0
            reasons = []
            
            if left_hand_up:
                confidence += 0.4
                reasons.append("Left hand raised")
            
            if right_hand_up:
                confidence += 0.4
                reasons.append("Right hand raised")
            
            # Both hands up significantly increases confidence
            if left_hand_up and right_hand_up:
                confidence += 0.2
                reasons.append("Both hands up")
            
            result['hands_up_confidence'] = min(confidence, 1.0)
            result['hands_up_reason'] = "; ".join(reasons) if reasons else "No hands up detected"
            
            # Check duration for confirmation
            current_time = time.time()
            
            if confidence > self.config.HANDS_UP_THRESHOLD:
                if self.hands_up_start_time is None:
                    self.hands_up_start_time = current_time
                
                duration = current_time - self.hands_up_start_time
                result['hands_up_duration'] = duration
                
                if duration >= self.config.HANDS_UP_DURATION_THRESHOLD:
                    result['hands_up_detected'] = True
                    if not self.hands_up_confirmed:
                        self.hands_up_confirmed = True
                        logger.warning(f"🚨 HANDS-UP DETECTED: Duration {duration:.1f}s")
            else:
                self.hands_up_start_time = None
                self.hands_up_confirmed = False
                
        except Exception as e:
            logger.error(f"Hands-up analysis error: {e}")
        
        return result
    
    def _analyze_remote_pickup(self, frame, detections, motion_detected, motion_score) -> Dict[str, Any]:
        """Analyze if remote pickup is occurring - simplified for immediate detection"""
        
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
        
        # Simplified pickup detection - if person and remote detected together, flag immediately
        if result['person_detected'] and result['remote_detected']:
            person_center = self._get_bbox_center(person_bbox)
            remote_center = self._get_bbox_center(remote_bbox)
            distance = self._calculate_distance(person_center, remote_center)
            result['distance_to_remote'] = distance
            
            # Immediate detection logic - much simpler
            confidence = 0.0
            reasons = []
            
            # If remote and person are detected in same frame, assume pickup
            if distance < self.config.PERSON_REMOTE_DISTANCE_THRESHOLD * 2:  # Doubled threshold for easier detection
                confidence = 0.8  # High confidence immediately
                reasons.append(f"Person with remote detected ({distance:.0f}px)")
                
                # Check if remote is in upper body area (likely in hand)
                person_height = person_bbox[3] - person_bbox[1]
                person_upper_body = person_bbox[1] + (person_height * 0.6)  # Upper 60% of person
                
                if remote_center[1] < person_upper_body:
                    confidence = 0.9
                    reasons.append("Remote in hand/upper body area")
                
                # Bonus for motion (but not required)
                if motion_detected:
                    confidence = min(confidence + 0.1, 1.0)
                    reasons.append("Motion detected")
            
            result['pickup_confidence'] = confidence
            result['pickup_reason'] = "; ".join(reasons) if reasons else "Person and remote detected"
            
            # Flag pickup if confidence > 0.5 (much lower threshold)
            if confidence > 0.5:
                result['pickup_detected'] = True
                logger.info(f"📺 REMOTE PICKUP: {reasons[0] if reasons else 'detected'}")
        
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
    
    def _record_hands_up_event(self, result):
        """Record hands-up event"""
        try:
            if len(self.frame_buffer) < 10:
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"hands_up_alert_{timestamp}.mp4"
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
            detection_stats['hands_up_alerts'] += 1
            
            logger.warning(f"🚨 HANDS-UP ALERT RECORDED: {filename} ({frames_written} frames)")
            
        except Exception as e:
            logger.error(f"Hands-up recording failed: {e}")
    
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
    
    def _draw_all_detections(self, frame, detections, pose_results, result) -> np.ndarray:
        """Draw all detection boxes and pose landmarks with clean minimal design"""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        try:
            # Draw YOLO detections with minimal boxes
            for _, detection in detections.iterrows():
                if detection['confidence'] > self.config.CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
                    name = detection['name']
                    conf = detection['confidence']
                    
                    # Clean minimal boxes
                    if name == 'person':
                        color = (0, 255, 0)  # Green
                    elif name in self.config.REMOTE_KEYWORDS:
                        color = (0, 100, 255)  # Orange-red
                    else:
                        color = (255, 255, 0)  # Cyan
                    
                    # Thin clean rectangle
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                    
                    # Clean minimal label
                    label = f"{name} {conf:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(annotated_frame, (x1, y1-label_h-5), (x1+label_w+5, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1+2, y1-3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw minimal pose landmarks
            if pose_results and pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                # Only draw key points, not full skeleton
                key_points = [
                    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
                ]
                
                for point in key_points:
                    landmark = landmarks[point]
                    if landmark.visibility > 0.5:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(annotated_frame, (x, y), 3, (255, 255, 255), -1)
                
                # Highlight raised hands with minimal indicators
                if result.get('left_hand_up', False):
                    left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
                    x, y = int(left_wrist.x * w), int(left_wrist.y * h)
                    cv2.circle(annotated_frame, (x, y), 8, (0, 0, 255), 2)
                
                if result.get('right_hand_up', False):
                    right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
                    x, y = int(right_wrist.x * w), int(right_wrist.y * h)
                    cv2.circle(annotated_frame, (x, y), 8, (0, 0, 255), 2)
            
            # Clean minimal status overlay in top-left corner
            overlay_y = 25
            
            if result.get('hands_up_detected', False):
                cv2.rectangle(annotated_frame, (5, 5), (200, 35), (0, 0, 255), -1)
                cv2.putText(annotated_frame, "HANDS UP ALERT", (10, overlay_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            elif result.get('pickup_detected', False):
                cv2.rectangle(annotated_frame, (5, 5), (180, 35), (0, 100, 255), -1)
                cv2.putText(annotated_frame, "REMOTE DETECTED", (10, overlay_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            elif result.get('hands_up_confidence', 0) > 0.3:
                duration = result.get('hands_up_duration', 0)
                cv2.rectangle(annotated_frame, (5, 5), (160, 35), (0, 165, 255), -1)
                cv2.putText(annotated_frame, f"HANDS RISING {duration:.1f}s", (10, overlay_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Minimal timestamp in bottom-right
            timestamp = datetime.now().strftime('%H:%M:%S')
            (text_w, text_h), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated_frame, (w-text_w-10, h-text_h-10), (w-5, h-5), (0, 0, 0), -1)
            cv2.putText(annotated_frame, timestamp, (w-text_w-7, h-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        except Exception as e:
            logger.error(f"Drawing error: {e}")
        
        return annotated_frame
    
    def _get_bbox_center(self, bbox) -> Tuple[int, int]:
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
    def _calculate_distance(self, point1, point2) -> float:
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            'person_detected': False,
            'remote_detected': False,
            'pickup_detected': False,
            'pickup_confidence': 0.0,
            'motion_score': 0.0,
            'distance_to_remote': None,
            'pickup_reason': '',
            'hands_up_detected': False,
            'hands_up_confidence': 0.0,
            'hands_up_duration': 0.0,
            'pose_detected': False,
            'left_hand_up': False,
            'right_hand_up': False,
            'hands_up_reason': '',
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
        annotated_frame, result = detector.detect_events(frame)
        
        # Update stats
        current_time = time.time()
        fps = frame_count / (current_time - start_time) if current_time > start_time else 0
        
        detection_stats.update({
            'person_detected': result['person_detected'],
            'remote_detected': result['remote_detected'],
            'pickup_detected': result['pickup_detected'],
            'pickup_confidence': result['pickup_confidence'],
            'hands_up_detected': result['hands_up_detected'],
            'hands_up_confidence': result['hands_up_confidence'],
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

# Enhanced HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Security Detection System</title>
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
        .alert-banner {
            background-color: #f44336;
            color: white;
            padding: 15px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: bold;
            display: none;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
        .alert-banner.hands-up { background-color: #d32f2f; }
        .alert-banner.monitoring { background-color: #ff9800; }
        .alert-banner.safe { background-color: #4CAF50; animation: none; }
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
        .stat-value.alert { color: #f44336; }
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
        .btn.emergency { background-color: #f44336; }
        .btn.emergency:hover { background-color: #d32f2f; }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-safe { background-color: #4CAF50; }
        .status-warning { background-color: #ff9800; }
        .status-danger { background-color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 Enhanced Security Detection System</h1>
            <p>TV Remote Pickup & Hands-Up Detection</p>
        </div>
        
        <div id="alert-banner" class="alert-banner safe">
            🟢 System Active - All Clear
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
                        <div>Remote Pickups</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value alert" id="hands-up-alerts">0</div>
                        <div>🚨 Hands-Up Alerts</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="fps">0</div>
                        <div>FPS</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="motion">0</div>
                        <div>Motion Level</div>
                    </div>
                </div>
                
                <div class="detection-details">
                    <h4>🎯 Current Detection</h4>
                    <p><span class="status-indicator status-safe" id="person-indicator"></span>Person: <span id="person-detected">No</span></p>
                    <p><span class="status-indicator status-safe" id="remote-indicator"></span>Remote: <span id="remote-detected">No</span></p>
                    <p><span class="status-indicator status-safe" id="pickup-indicator"></span>Pickup: <span id="pickup-confidence">0%</span></p>
                </div>
                
                <div class="detection-details">
                    <h4>🚨 Hands-Up Detection</h4>
                    <p><span class="status-indicator status-safe" id="pose-indicator"></span>Pose: <span id="pose-detected">No</span></p>
                    <p><span class="status-indicator status-safe" id="left-hand-indicator"></span>Left Hand: <span id="left-hand-up">Down</span></p>
                    <p><span class="status-indicator status-safe" id="right-hand-indicator"></span>Right Hand: <span id="right-hand-up">Down</span></p>
                    <p><span class="status-indicator status-safe" id="hands-up-indicator"></span>Alert Level: <span id="hands-up-confidence">0%</span></p>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="location.reload()">🔄 Refresh</button>
            <button class="btn" onclick="captureFrame()">📷 Capture</button>
            <button class="btn emergency" onclick="emergencyAlert()">🚨 Emergency Alert</button>
            <button class="btn" onclick="viewRecordings()">📹 View Recordings</button>
        </div>
        
        <div style="margin-top: 30px; padding: 20px; background-color: #2a2a2a; border-radius: 10px;">
            <h4>ℹ️ System Information</h4>
            <p><strong>Detection Capabilities:</strong></p>
            <ul>
                <li>📺 TV Remote pickup detection using YOLO object detection</li>
                <li>🚨 Hands-up detection using MediaPipe pose estimation</li>
                <li>🎯 Motion detection and tracking</li>
                <li>📹 Automatic event recording</li>
            </ul>
            <p><strong>Security Features:</strong></p>
            <ul>
                <li>Real-time pose analysis for potential hostage situations</li>
                <li>Configurable sensitivity and duration thresholds</li>
                <li>Automatic video recording of security events</li>
                <li>Web-based monitoring interface</li>
            </ul>
        </div>
    </div>
    
    <script>
        let alertSound = null;
        let lastHandsUpAlert = false;
        
        // Initialize audio for alerts
        function initAudio() {
            try {
                alertSound = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigELIHO8diJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+PTrm0gBSuVyfPBbigE=');
            } catch (e) {
                console.log('Audio initialization failed');
            }
        }
        
        setInterval(updateStats, 500);
        
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    // Update basic stats
                    document.getElementById('total-pickups').textContent = data.remote_pickups || 0;
                    document.getElementById('hands-up-alerts').textContent = data.hands_up_alerts || 0;
                    document.getElementById('fps').textContent = data.fps || 0;
                    document.getElementById('motion').textContent = Math.round(data.motion_score || 0);
                    
                    // Update detection status
                    updateIndicator('person-indicator', data.person_detected);
                    updateIndicator('remote-indicator', data.remote_detected);
                    updateIndicator('pickup-indicator', data.pickup_detected);
                    updateIndicator('pose-indicator', data.pose_detected);
                    updateIndicator('left-hand-indicator', data.left_hand_up);
                    updateIndicator('right-hand-indicator', data.right_hand_up);
                    updateIndicator('hands-up-indicator', data.hands_up_detected);
                    
                    document.getElementById('person-detected').textContent = data.person_detected ? 'Yes' : 'No';
                    document.getElementById('remote-detected').textContent = data.remote_detected ? 'Yes' : 'No';
                    document.getElementById('pickup-confidence').textContent = Math.round((data.pickup_confidence || 0) * 100) + '%';
                    document.getElementById('pose-detected').textContent = data.pose_detected ? 'Yes' : 'No';
                    document.getElementById('left-hand-up').textContent = data.left_hand_up ? 'Up' : 'Down';
                    document.getElementById('right-hand-up').textContent = data.right_hand_up ? 'Up' : 'Down';
                    document.getElementById('hands-up-confidence').textContent = Math.round((data.hands_up_confidence || 0) * 100) + '%';
                    
                    // Update main alert banner
                    const alertBanner = document.getElementById('alert-banner');
                    
                    if (data.hands_up_detected) {
                        alertBanner.textContent = '🚨 HANDS UP DETECTED - POTENTIAL EMERGENCY! 🚨';
                        alertBanner.className = 'alert-banner hands-up';
                        alertBanner.style.display = 'block';
                        
                        // Play alert sound for hands-up
                        if (!lastHandsUpAlert && alertSound) {
                            alertSound.play().catch(e => console.log('Audio play failed'));
                        }
                        lastHandsUpAlert = true;
                    } else if (data.pickup_detected) {
                        alertBanner.textContent = '📺 Remote in Hand Detected!';
                        alertBanner.className = 'alert-banner';
                        alertBanner.style.display = 'block';
                        lastHandsUpAlert = false;
                    } else if (data.person_detected && data.remote_detected) {
                        alertBanner.textContent = '👀 Person + Remote Detected';
                        alertBanner.className = 'alert-banner monitoring';
                        alertBanner.style.display = 'block';
                        lastHandsUpAlert = false;
                    } else {
                        alertBanner.textContent = '🟢 System Active - All Clear';
                        alertBanner.className = 'alert-banner safe';
                        alertBanner.style.display = 'block';
                        lastHandsUpAlert = false;
                    }
                })
                .catch(err => console.error('Stats update failed:', err));
        }
        
        function updateIndicator(elementId, isActive) {
            const indicator = document.getElementById(elementId);
            if (isActive) {
                indicator.className = 'status-indicator status-danger';
            } else {
                indicator.className = 'status-indicator status-safe';
            }
        }
        
        function captureFrame() {
            const link = document.createElement('a');
            link.href = '/capture';
            link.download = 'security_capture_' + Date.now() + '.jpg';
            link.click();
        }
        
        function emergencyAlert() {
            if (confirm('Send emergency alert? This will notify security personnel.')) {
                fetch('/emergency_alert', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        alert('Emergency alert sent! Response: ' + data.message);
                    })
                    .catch(err => {
                        alert('Emergency alert failed to send!');
                        console.error(err);
                    });
            }
        }
        
        function viewRecordings() {
            window.open('/recordings', '_blank');
        }
        
        // Initialize on page load
        initAudio();
        updateStats();
    </script>
</body>
</html>
"""

# Flask Routes (Enhanced)
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

@app.route('/emergency_alert', methods=['POST'])
def emergency_alert():
    """Handle manual emergency alerts"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.warning(f"🚨 MANUAL EMERGENCY ALERT TRIGGERED at {timestamp}")
        
        global detection_stats
        detection_stats['hands_up_alerts'] += 1
        
        return jsonify({
            "status": "success",
            "message": f"Emergency alert logged at {timestamp}",
            "timestamp": timestamp
        })
    except Exception as e:
        logger.error(f"Emergency alert failed: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to process emergency alert"
        }), 500

@app.route('/recordings')
def recordings():
    """List recorded events"""
    try:
        files = []
        if os.path.exists(Config.OUTPUT_DIR):
            for filename in sorted(os.listdir(Config.OUTPUT_DIR)):
                if filename.endswith(('.mp4', '.avi')):
                    filepath = os.path.join(Config.OUTPUT_DIR, filename)
                    size = os.path.getsize(filepath)
                    mtime = os.path.getmtime(filepath)
                    files.append({
                        'filename': filename,
                        'size': f"{size / (1024*1024):.1f} MB",
                        'date': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'Hands-Up Alert' if 'hands_up' in filename else 'Remote Pickup'
                    })
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Recorded Events</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #1a1a1a; color: white; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #333; }}
                tr:nth-child(even) {{ background-color: #2a2a2a; }}
                .btn {{ background-color: #4CAF50; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; }}
                .alert {{ color: #f44336; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📹 Recorded Security Events</h1>
                <p>Total recordings: {len(files)}</p>
                
                <table>
                    <tr>
                        <th>Event Type</th>
                        <th>Filename</th>
                        <th>Date</th>
                        <th>Size</th>
                        <th>Action</th>
                    </tr>
        """
        
        for file_info in files:
            event_class = 'alert' if 'Hands-Up' in file_info['type'] else ''
            html += f"""
                    <tr>
                        <td class="{event_class}">{file_info['type']}</td>
                        <td>{file_info['filename']}</td>
                        <td>{file_info['date']}</td>
                        <td>{file_info['size']}</td>
                        <td><a href="/download/{file_info['filename']}" class="btn">Download</a></td>
                    </tr>
            """
        
        html += """
                </table>
                <br>
                <a href="/" class="btn">← Back to Live Feed</a>
            </div>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        return f"Error loading recordings: {e}", 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download recorded files"""
    try:
        filepath = os.path.join(Config.OUTPUT_DIR, filename)
        if os.path.exists(filepath) and filename.endswith(('.mp4', '.avi')):
            return Response(
                open(filepath, 'rb').read(),
                mimetype='video/mp4',
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
        else:
            return "File not found", 404
    except Exception as e:
        return f"Download error: {e}", 500

def initialize_system():
    """Initialize camera and detector"""
    global camera, detector
    
    try:
        logger.info("🚀 Initializing Enhanced Security Detection System...")
        
        config = Config()
        camera = LibCameraCapture(config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS)
        
        if not camera.start():
            logger.error("❌ Failed to start camera")
            return False
        
        detector = EnhancedDetector(config)
        
        logger.info("✅ System initialized successfully")
        logger.info("📺 Remote pickup detection: ENABLED")
        logger.info("🚨 Hands-up detection: ENABLED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False

def main():
    logger.info("🔒 Starting Enhanced Security Detection System")
    logger.info("Features: TV Remote Pickup + Hands-Up Detection")
    
    if not initialize_system():
        logger.error("❌ System initialization failed")
        return False
    
    try:
        logger.info(f"🌐 Web interface: http://localhost:{Config.WEB_PORT}")
        logger.info("🚨 Security monitoring active")
        logger.info("Press Ctrl+C to stop")
        
        app.run(
            host='0.0.0.0',
            port=Config.WEB_PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Stopping security system...")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        if camera:
            camera.stop()
        logger.info("✅ Security system shutdown complete")

if __name__ == "__main__":
    main()
