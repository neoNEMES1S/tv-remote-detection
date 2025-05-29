#!/usr/bin/env python3
"""
Advanced TV Remote Pickup Detection System with YOLOv11
Enhanced low-light performance and multi-strategy remote detection
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
from typing import List, Optional, Tuple, Dict, Any
from flask import Flask, render_template_string, Response, jsonify
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Config:
    """Configuration for TV Remote Detection System"""
    
    # Camera settings - optimized for low light
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 10
    CAMERA_EXPOSURE: int = 15000  # Microseconds
    CAMERA_ANALOGUE_GAIN: float = 12.0  # Higher for low light
    CAMERA_DENOISE: str = "cdn_hq"  # High quality denoise
    
    # AI Model settings
    YOLO_MODEL: str = 'yolov11n.pt'  # YOLOv11 nano
    CONFIDENCE_THRESHOLD: float = 0.15  # Lower for better detection
    NMS_THRESHOLD: float = 0.45
    PROCESSING_RESOLUTION: Tuple[int, int] = (320, 320)
    
    # Performance settings
    DETECTION_SKIP_FRAMES: int = 2  # Process every 2nd frame
    USE_GPU: bool = False
    
    # Remote detection settings - Multi-strategy
    ENABLE_COLOR_DETECTION: bool = True
    ENABLE_CONTOUR_DETECTION: bool = True
    ENABLE_EDGE_DETECTION: bool = True
    
    # Object keywords for YOLO
    REMOTE_KEYWORDS: List[str] = ['remote', 'cell phone', 'book', 'remote control', 
                                  'smartphone', 'device', 'controller']
    HAND_KEYWORDS: List[str] = ['person', 'hand']
    
    # Detection thresholds
    PERSON_REMOTE_DISTANCE_THRESHOLD: int = 200
    REMOTE_SIZE_RANGE: Tuple[int, int] = (15, 250)  # Min/max width
    REMOTE_ASPECT_RATIO_RANGE: Tuple[float, float] = (0.2, 6.0)
    
    # Color detection for black remotes
    BLACK_LOWER_HSV: List[int] = [0, 0, 0]
    BLACK_UPPER_HSV: List[int] = [180, 255, 50]
    
    # Motion sensitivity
    MOTION_THRESHOLD: int = 300
    PICKUP_MOTION_THRESHOLD: int = 600
    MOTION_BLUR_SIZE: int = 21
    
    # Recording settings
    CLIP_DURATION_SECONDS: int = 6
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
    "remote_detection_method": "",
    "pickup_detected": False,
    "fps": 0,
    "actual_fps": 0,
    "inference_time": 0,
    "timestamp": ""
}

app = Flask(__name__)

class EnhancedLibCameraCapture:
    """Enhanced camera capture with low-light optimization"""
    def __init__(self, config: Config):
        self.config = config
        self.frame_queue = queue.Queue(maxsize=2)
        self.process = None
        self.capture_thread = None
        self.running = False
        self.frame_count = 0
        
    def start(self) -> bool:
        """Start camera with enhanced low-light settings"""
        try:
            self._stop()
            
            # Enhanced libcamera command for low light
            cmd = [
                'libcamera-vid',
                '--width', str(self.config.CAMERA_WIDTH),
                '--height', str(self.config.CAMERA_HEIGHT),
                '--framerate', str(self.config.CAMERA_FPS),
                '--timeout', '0',
                '--codec', 'yuv420',
                '--output', '-',
                '--nopreview',
                '--flush',
                # Low light optimizations
                '--shutter', str(self.config.CAMERA_EXPOSURE),
                '--analoggain', str(self.config.CAMERA_ANALOGUE_GAIN),
                '--denoise', self.config.CAMERA_DENOISE,
                '--awb', 'auto',  # Auto white balance
                '--brightness', '0.1',  # Slight brightness boost
                '--contrast', '1.2',  # Slight contrast boost
                '--sharpness', '1.5'  # Sharpness for edge detection
            ]
            
            logger.info("Starting camera with low-light optimizations...")
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0, preexec_fn=os.setsid
            )
            
            time.sleep(3)  # Extra time for camera adjustment
            
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
                        logger.info(f"✓ Camera started with low-light mode: {test_frame.shape}")
                        return True
                time.sleep(0.1)
            
            logger.error("No frames received")
            return False
            
        except Exception as e:
            logger.error(f"Camera start failed: {e}")
            return False
    
    def _capture_frames(self):
        """Capture and pre-process frames"""
        frame_size = self.config.CAMERA_WIDTH * self.config.CAMERA_HEIGHT * 3 // 2
        bytes_buffer = b""
        
        logger.info("Enhanced frame capture started")
        
        while self.running and self.process and self.process.poll() is None:
            try:
                chunk = self.process.stdout.read(8192)
                if not chunk:
                    time.sleep(0.001)
                    continue
                
                bytes_buffer += chunk
                
                while len(bytes_buffer) >= frame_size:
                    frame_data = bytes_buffer[:frame_size]
                    bytes_buffer = bytes_buffer[frame_size:]
                    
                    try:
                        yuv_data = np.frombuffer(frame_data, dtype=np.uint8)
                        yuv_frame = yuv_data.reshape((self.config.CAMERA_HEIGHT * 3 // 2, self.config.CAMERA_WIDTH))
                        bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                        
                        # Apply low-light enhancement
                        bgr_frame = self._enhance_low_light_frame(bgr_frame)
                        
                        # Drop old frames
                        while not self.frame_queue.empty():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                break
                        
                        self.frame_queue.put(bgr_frame, block=False)
                        self.frame_count += 1
                        
                    except Exception as e:
                        logger.error(f"Frame processing error: {e}")
                        
            except Exception:
                time.sleep(0.001)
    
    def _enhance_low_light_frame(self, frame):
        """Apply low-light enhancement to frame"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Slight denoising
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            return enhanced
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            return frame
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frame = self.frame_queue.get(timeout=0.1)
            return True, frame
        except queue.Empty:
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
    
    def stop(self):
        self._stop()

class MultiStrategyRemoteDetector:
    """Advanced remote detector with multiple detection strategies"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.previous_frame_gray = None
        self.motion_history = deque(maxlen=5)
        self.frame_buffer = deque(maxlen=config.CLIP_DURATION_SECONDS * config.CAMERA_FPS)
        self.frame_count = 0
        self.remote_candidates_history = deque(maxlen=10)
        self._load_model()
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
    def _load_model(self):
        """Load YOLOv11 model"""
        try:
            logger.info("Loading YOLOv11 model...")
            
            # Try to load YOLOv11
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov11n.pt')  # Will download if not present
                logger.info("✓ YOLOv11 loaded successfully")
            except Exception as e:
                logger.warning(f"YOLOv11 not available, falling back to YOLOv5: {e}")
                # Fallback to YOLOv5
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
                self.model.conf = self.config.CONFIDENCE_THRESHOLD
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def detect_remote_pickup(self, frame) -> Tuple[np.ndarray, Dict]:
        """Multi-strategy detection function"""
        if frame is None or frame.size == 0:
            return frame, self._empty_result()
        
        try:
            self.frame_count += 1
            self.frame_buffer.append(frame.copy())
            
            # Motion detection
            motion_detected, motion_score, motion_region = self._detect_motion_optimized(frame)
            
            # Skip detection based on frame count
            if self.frame_count % self.config.DETECTION_SKIP_FRAMES != 0:
                result = self._get_cached_result(motion_detected, motion_score)
                annotated_frame = self._draw_cached_detections(frame, result)
                return annotated_frame, result
            
            # Multi-strategy detection
            inference_start = time.time()
            
            # Strategy 1: YOLO detection
            yolo_detections = self._run_yolo_detection(frame, motion_region)
            
            # Strategy 2: Color-based detection (for black remotes)
            color_candidates = []
            if self.config.ENABLE_COLOR_DETECTION:
                color_candidates = self._detect_by_color(frame)
            
            # Strategy 3: Contour detection
            contour_candidates = []
            if self.config.ENABLE_CONTOUR_DETECTION:
                contour_candidates = self._detect_by_contours(frame)
            
            # Strategy 4: Edge detection
            edge_candidates = []
            if self.config.ENABLE_EDGE_DETECTION:
                edge_candidates = self._detect_by_edges(frame)
            
            # Merge all detection strategies
            all_candidates = self._merge_detections(
                yolo_detections, color_candidates, contour_candidates, edge_candidates
            )
            
            inference_time = (time.time() - inference_start) * 1000
            
            # Analyze for pickup
            result = self._analyze_pickup_advanced(frame, all_candidates, motion_detected, motion_score)
            result['inference_time'] = inference_time
            
            # Update cache
            self._update_cache(all_candidates, result)
            
            # Create annotated frame
            annotated_frame = self._draw_advanced_detections(frame, all_candidates, result)
            
            return annotated_frame, result
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return frame, self._empty_result()
    
    def _run_yolo_detection(self, frame, motion_region):
        """Run YOLO detection with YOLOv11 or v5"""
        try:
            # Crop to motion region if available
            if motion_region:
                x1, y1, x2, y2 = motion_region
                crop_frame = frame[y1:y2, x1:x2]
                offset = (x1, y1)
            else:
                crop_frame = frame
                offset = (0, 0)
            
            # Resize for model
            h, w = crop_frame.shape[:2]
            new_w, new_h = self.config.PROCESSING_RESOLUTION
            frame_resized = cv2.resize(crop_frame, (new_w, new_h))
            
            # Run detection based on model type
            detections = []
            
            if hasattr(self.model, 'predict'):  # YOLOv11
                results = self.model.predict(
                    frame_resized,
                    conf=self.config.CONFIDENCE_THRESHOLD,
                    iou=self.config.NMS_THRESHOLD,
                    verbose=False
                )
                
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = box.conf[0].item()
                            cls = int(box.cls[0].item())
                            name = self.model.names[cls]
                            
                            # Scale back
                            x1 = int(x1 * w / new_w) + offset[0]
                            y1 = int(y1 * h / new_h) + offset[1]
                            x2 = int(x2 * w / new_w) + offset[0]
                            y2 = int(y2 * h / new_h) + offset[1]
                            
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'name': name,
                                'source': 'yolo'
                            })
            else:  # YOLOv5
                results = self.model(frame_resized)
                pandas_detections = results.pandas().xyxy[0]
                
                for _, det in pandas_detections.iterrows():
                    x1 = int(det['xmin'] * w / new_w) + offset[0]
                    y1 = int(det['ymin'] * h / new_h) + offset[1]
                    x2 = int(det['xmax'] * w / new_w) + offset[0]
                    y2 = int(det['ymax'] * h / new_h) + offset[1]
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': det['confidence'],
                        'name': det['name'],
                        'source': 'yolo'
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def _detect_by_color(self, frame):
        """Detect black remote-like objects by color"""
        try:
            candidates = []
            
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for black objects
            lower = np.array(self.config.BLACK_LOWER_HSV)
            upper = np.array(self.config.BLACK_UPPER_HSV)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100 or area > 10000:  # Filter by area
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check if remote-like
                if (self.config.REMOTE_SIZE_RANGE[0] < w < self.config.REMOTE_SIZE_RANGE[1] and
                    self.config.REMOTE_ASPECT_RATIO_RANGE[0] < aspect_ratio < self.config.REMOTE_ASPECT_RATIO_RANGE[1]):
                    
                    candidates.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.5,  # Medium confidence for color detection
                        'name': 'remote_color',
                        'source': 'color'
                    })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Color detection error: {e}")
            return []
    
    def _detect_by_contours(self, frame):
        """Detect rectangular objects by contours"""
        try:
            candidates = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Bilateral filter to reduce noise while keeping edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Look for rectangular shapes (4 vertices)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Check if remote-like
                    if (self.config.REMOTE_SIZE_RANGE[0] < w < self.config.REMOTE_SIZE_RANGE[1] and
                        self.config.REMOTE_ASPECT_RATIO_RANGE[0] < aspect_ratio < self.config.REMOTE_ASPECT_RATIO_RANGE[1]):
                        
                        candidates.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': 0.4,  # Lower confidence for contour detection
                            'name': 'remote_contour',
                            'source': 'contour'
                        })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Contour detection error: {e}")
            return []
    
    def _detect_by_edges(self, frame):
        """Detect objects using edge detection"""
        try:
            candidates = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate edges to connect nearby edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours from edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200 or area > 8000:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check rectangularity
                rect_area = w * h
                extent = area / rect_area if rect_area > 0 else 0
                
                if (self.config.REMOTE_SIZE_RANGE[0] < w < self.config.REMOTE_SIZE_RANGE[1] and
                    self.config.REMOTE_ASPECT_RATIO_RANGE[0] < aspect_ratio < self.config.REMOTE_ASPECT_RATIO_RANGE[1] and
                    extent > 0.7):  # Object fills most of bounding box
                    
                    candidates.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.35,  # Lower confidence for edge detection
                        'name': 'remote_edge',
                        'source': 'edge'
                    })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Edge detection error: {e}")
            return []
    
    def _merge_detections(self, yolo_dets, color_dets, contour_dets, edge_dets):
        """Merge detections from all strategies with NMS"""
        all_detections = []
        
        # Add all detections
        all_detections.extend(yolo_dets)
        all_detections.extend(color_dets)
        all_detections.extend(contour_dets)
        all_detections.extend(edge_dets)
        
        if not all_detections:
            return []
        
        # Apply Non-Maximum Suppression
        boxes = np.array([d['bbox'] for d in all_detections])
        scores = np.array([d['confidence'] for d in all_detections])
        
        # Simple NMS implementation
        indices = self._non_max_suppression(boxes, scores, self.config.NMS_THRESHOLD)
        
        # Keep only selected detections
        merged = [all_detections[i] for i in indices]
        
        # Update remote candidates history
        self.remote_candidates_history.append(merged)
        
        return merged
    
    def _non_max_suppression(self, boxes, scores, threshold):
        """Simple NMS implementation"""
        if len(boxes) == 0:
            return []
        
        # Convert to x1, y1, x2, y2 format if needed
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _detect_motion_optimized(self, frame) -> Tuple[bool, float, Optional[Tuple]]:
        """Enhanced motion detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (self.config.MOTION_BLUR_SIZE, self.config.MOTION_BLUR_SIZE), 0)
            
            if self.previous_frame_gray is None:
                self.previous_frame_gray = gray
                return False, 0, None
            
            # Frame difference
            diff = cv2.absdiff(gray, self.previous_frame_gray)
            _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            
            # Dilate to connect nearby regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            
            # Find motion region
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_region = None
            if contours:
                # Get bounding box of all motion
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = 0, 0
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)
                
                # Add padding
                padding = 50
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame.shape[1], x_max + padding)
                y_max = min(frame.shape[0], y_max + padding)
                
                motion_region = (x_min, y_min, x_max, y_max)
            
            motion_score = np.sum(thresh) / 255
            self.motion_history.append(motion_score)
            
            # Use average motion for stability
            avg_motion = np.mean(self.motion_history) if self.motion_history else 0
            
            self.previous_frame_gray = gray
            return avg_motion > self.config.MOTION_THRESHOLD, motion_score, motion_region
            
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return False, 0, None
    
    def _analyze_pickup_advanced(self, frame, detections, motion_detected, motion_score) -> Dict:
        """Advanced pickup analysis with multi-strategy support"""
        
        result = {
            'person_detected': False,
            'remote_detected': False,
            'remote_detection_method': '',
            'pickup_detected': False,
            'pickup_confidence': 0.0,
            'motion_score': motion_score,
            'distance_to_remote': None,
            'pickup_reason': '',
            'timestamp': datetime.now().isoformat()
        }
        
        if not detections:
            return result
        
        # Find person
        person_detections = [d for d in detections if d['name'] == 'person' and d['source'] == 'yolo']
        person_bbox = None
        
        if person_detections:
            result['person_detected'] = True
            # Get largest person (likely closest)
            person = max(person_detections, 
                        key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))
            person_bbox = person['bbox']
        
        # Find remote candidates from all sources
        remote_candidates = []
        detection_methods = set()
        
        for det in detections:
            bbox = det['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = width / height if height > 0 else 0
            
            # Check if it's a remote-like object
            is_remote_keyword = det['name'] in self.config.REMOTE_KEYWORDS
            is_remote_detection = 'remote' in det['name']
            is_remote_size = (self.config.REMOTE_SIZE_RANGE[0] < width < self.config.REMOTE_SIZE_RANGE[1])
            is_remote_shape = (self.config.REMOTE_ASPECT_RATIO_RANGE[0] < aspect_ratio < 
                              self.config.REMOTE_ASPECT_RATIO_RANGE[1])
            
            if (is_remote_keyword or is_remote_detection or 
                (is_remote_size and is_remote_shape and det['confidence'] > 0.2)):
                
                remote_candidates.append({
                    'bbox': bbox,
                    'confidence': det['confidence'],
                    'name': det['name'],
                    'source': det['source'],
                    'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                })
                detection_methods.add(det['source'])
        
        # Use best remote candidate
        if remote_candidates:
            result['remote_detected'] = True
            result['remote_detection_method'] = ', '.join(detection_methods)
            
            # Prefer higher confidence and YOLO detections
            remote_candidates.sort(key=lambda r: (
                1.0 if r['source'] == 'yolo' else 0.5,  # Prefer YOLO
                r['confidence']
            ), reverse=True)
            
            # If person detected, prefer remotes near person
            if person_bbox:
                hand_y = person_bbox[1] + (person_bbox[3] - person_bbox[1]) * 0.5
                
                # Re-sort by proximity to person
                for candidate in remote_candidates:
                    distance_to_person = abs(candidate['center'][1] - hand_y)
                    candidate['person_proximity_score'] = 1.0 / (1.0 + distance_to_person / 100)
                
                remote_candidates.sort(
                    key=lambda r: r['confidence'] * r.get('person_proximity_score', 0.5),
                    reverse=True
                )
            
            remote = remote_candidates[0]
            remote_bbox = remote['bbox']
        
        # Enhanced pickup detection
        if result['person_detected'] and result['remote_detected']:
            person_center = self._get_bbox_center(person_bbox)
            remote_center = remote['center']
            distance = self._calculate_distance(person_center, remote_center)
            result['distance_to_remote'] = distance
            
            # Sophisticated pickup logic
            confidence = 0.0
            reasons = []
            
            # 1. Proximity check with scaled threshold
            proximity_threshold = self.config.PERSON_REMOTE_DISTANCE_THRESHOLD
            if distance < proximity_threshold:
                proximity_score = 1.0 - (distance / proximity_threshold)
                confidence += 0.3 * proximity_score
                reasons.append(f"Proximity ({distance:.0f}px)")
            
            # 2. Motion correlation
            if motion_detected:
                motion_weight = min(motion_score / self.config.PICKUP_MOTION_THRESHOLD, 1.0)
                confidence += 0.25 * motion_weight
                reasons.append(f"Motion ({motion_score:.0f})")
            
            # 3. Remote in hand area
            hand_area_start = person_bbox[1] + (person_bbox[3] - person_bbox[1]) * 0.3
            hand_area_end = person_bbox[1] + (person_bbox[3] - person_bbox[1]) * 0.8
            
            if hand_area_start < remote_center[1] < hand_area_end:
                confidence += 0.2
                reasons.append("Hand area")
            
            # 4. Remote overlaps with person
            overlap = self._calculate_overlap(person_bbox, remote_bbox)
            if overlap > 0:
                confidence += 0.15 * min(overlap / 0.3, 1.0)
                reasons.append(f"Overlap ({overlap:.0%})")
            
            # 5. Boost confidence based on detection method
            if remote['source'] == 'yolo':
                confidence += 0.1
                reasons.append("YOLO detection")
            elif len(detection_methods) > 1:
                confidence += 0.05
                reasons.append("Multi-method")
            
            result['pickup_confidence'] = min(confidence, 1.0)
            result['pickup_reason'] = "; ".join(reasons) if reasons else "No indicators"
            
            # Adjusted threshold for better detection
            if confidence > 0.45:  # Lower threshold
                result['pickup_detected'] = True
                self._record_pickup_event(result)
        
        return result
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate IoU overlap between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_cached_result(self, motion_detected, motion_score) -> Dict:
        """Get cached result for skipped frames"""
        result = self._empty_result()
        result['motion_score'] = motion_score
        
        if hasattr(self, '_last_result'):
            result.update({
                'person_detected': self._last_result.get('person_detected', False),
                'remote_detected': self._last_result.get('remote_detected', False),
                'remote_detection_method': self._last_result.get('remote_detection_method', ''),
                'pickup_detected': self._last_result.get('pickup_detected', False),
                'pickup_confidence': self._last_result.get('pickup_confidence', 0.0),
                'distance_to_remote': self._last_result.get('distance_to_remote')
            })
        
        return result
    
    def _update_cache(self, detections, result):
        """Update detection cache"""
        self._last_detections = detections.copy() if detections else []
        self._last_result = result.copy()
    
    def _draw_cached_detections(self, frame, result) -> np.ndarray:
        """Draw using cached detections"""
        if hasattr(self, '_last_detections'):
            return self._draw_advanced_detections(frame, self._last_detections, result)
        return self._draw_status_only(frame, result)
    
    def _draw_status_only(self, frame, result) -> np.ndarray:
        """Draw only status overlay"""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Status overlay
        if result['pickup_detected']:
            cv2.putText(annotated_frame, "PICKUP DETECTED!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif result['person_detected'] and result['remote_detected']:
            cv2.putText(annotated_frame, "MONITORING...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return annotated_frame
    
    def _draw_advanced_detections(self, frame, detections, result) -> np.ndarray:
        """Draw detections with source information"""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        try:
            # Color scheme for different sources
            source_colors = {
                'yolo': (0, 255, 0),      # Green
                'color': (255, 0, 0),      # Blue
                'contour': (255, 255, 0),  # Cyan
                'edge': (255, 0, 255),     # Magenta
            }
            
            # Draw all detections
            for det in detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                source = det['source']
                name = det['name']
                conf = det['confidence']
                
                color = source_colors.get(source, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Label with source info
                label = f"{name}[{source}]: {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Special highlight for remote candidates
                if 'remote' in name.lower():
                    cv2.putText(annotated_frame, "REMOTE?", (x1, y1-25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw status overlay with gradient background
            overlay = annotated_frame.copy()
            status_height = 70
            
            if result['pickup_detected']:
                cv2.rectangle(overlay, (0, 0), (w, status_height), (0, 0, 100), -1)
                cv2.putText(overlay, "REMOTE PICKUP DETECTED!", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            elif result['person_detected'] and result['remote_detected']:
                cv2.rectangle(overlay, (0, 0), (w, status_height), (100, 50, 0), -1)
                cv2.putText(overlay, f"MONITORING... Confidence: {result['pickup_confidence']:.0%}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(overlay, f"Method: {result['remote_detection_method']}", 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.rectangle(overlay, (0, 0), (w, status_height), (50, 50, 0), -1)
                cv2.putText(overlay, "Waiting for activity...", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            # Info bar at bottom
            info_height = 50
            cv2.rectangle(annotated_frame, (0, h-info_height), (w, h), (30, 30, 30), -1)
            
            # Stats
            cv2.putText(annotated_frame, f"FPS: {detection_stats.get('actual_fps', 0):.1f}", 
                       (10, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Motion: {result['motion_score']:.0f}", 
                       (150, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Inf: {result.get('inference_time', 0):.0f}ms", 
                       (300, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Timestamp
            timestamp = datetime.now().strftime('%H:%M:%S')
            cv2.putText(annotated_frame, timestamp, (w - 100, h-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Detection count
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", 
                       (450, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        except Exception as e:
            logger.error(f"Drawing error: {e}")
        
        return annotated_frame
    
    def _record_pickup_event(self, result):
        """Record pickup event with metadata"""
        try:
            if len(self.frame_buffer) < self.config.CAMERA_FPS:
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"remote_pickup_{timestamp}.mp4"
            output_path = os.path.join(self.config.OUTPUT_DIR, filename)
            
            # Use H264 codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.config.CAMERA_FPS, 
                                (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
            
            frames_written = 0
            for frame in self.frame_buffer:
                if frame is not None:
                    out.write(frame)
                    frames_written += 1
            
            out.release()
            
            # Save detailed metadata
            metadata = {
                'timestamp': result['timestamp'],
                'confidence': result['pickup_confidence'],
                'reason': result['pickup_reason'],
                'detection_method': result['remote_detection_method'],
                'motion_score': result['motion_score'],
                'distance': result['distance_to_remote'],
                'frames': frames_written
            }
            
            with open(output_path.replace('.mp4', '.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            global detection_stats
            detection_stats['remote_pickups'] += 1
            
            logger.info(f"📹 Recorded: {filename} ({frames_written} frames, "
                       f"{result['pickup_confidence']:.2f} conf, {result['remote_detection_method']})")
            
        except Exception as e:
            logger.error(f"Recording failed: {e}")
    
    def _get_bbox_center(self, bbox) -> Tuple[int, int]:
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
    def _calculate_distance(self, point1, point2) -> float:
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _empty_result(self) -> Dict:
        return {
            'person_detected': False,
            'remote_detected': False,
            'remote_detection_method': '',
            'pickup_detected': False,
            'pickup_confidence': 0.0,
            'motion_score': 0.0,
            'distance_to_remote': None,
            'pickup_reason': '',
            'inference_time': 0,
            'timestamp': datetime.now().isoformat()
        }

def generate_frames():
    """Generate frames for web streaming"""
    global current_frame, detection_stats
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    last_fps_update = time.time()
    
    while True:
        if camera is None or detector is None:
            time.sleep(0.1)
            continue
            
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        fps_counter += 1
        
        # Run detection
        annotated_frame, result = detector.detect_remote_pickup(frame)
        
        # Update FPS
        current_time = time.time()
        if current_time - last_fps_update >= 1.0:
            actual_fps = fps_counter / (current_time - last_fps_update)
            fps_counter = 0
            last_fps_update = current_time
            
            overall_fps = frame_count / (current_time - fps_start_time) if current_time > fps_start_time else 0
            
            detection_stats.update({
                'actual_fps': round(actual_fps, 1),
                'fps': round(overall_fps, 1)
            })
        
        # Update stats
        detection_stats.update({
            'person_detected': result['person_detected'],
            'remote_detected': result['remote_detected'],
            'remote_detection_method': result['remote_detection_method'],
            'pickup_detected': result['pickup_detected'],
            'pickup_confidence': result['pickup_confidence'],
            'motion_score': result['motion_score'],
            'inference_time': result.get('inference_time', 0),
            'frame_count': frame_count,
            'timestamp': result['timestamp']
        })
        
        current_frame = annotated_frame
        
        # Encode frame
        try:
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 70]
            ret, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
        
        time.sleep(0.01)

# Enhanced HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>TV Remote Detection - YOLOv11 Multi-Strategy</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #0a0a0a; 
            color: white; 
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
        }
        .header { 
            text-align: center; 
            margin-bottom: 20px; 
            background: linear-gradient(45deg, #1a1a1a, #2a2a2a);
            padding: 20px;
            border-radius: 10px;
        }
        .main-content { 
            display: flex; 
            gap: 20px; 
            margin-bottom: 20px; 
        }
        .video-section { 
            flex: 2; 
            background-color: #1a1a1a;
            padding: 10px;
            border-radius: 10px;
        }
        .stats-section { 
            flex: 1; 
            background-color: #1a1a1a; 
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
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .pickup-alert.monitoring { 
            background: linear-gradient(45deg, #ff6b00, #ff9800);
            animation: pulse 2s infinite;
        }
        .pickup-alert.safe { 
            background: linear-gradient(45deg, #2e7d32, #4CAF50);
        }
        
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.9; transform: scale(1.02); }
            100% { opacity: 1; transform: scale(1); }
        }
        
        .stats-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 15px; 
            margin-bottom: 20px; 
        }
        .stat-card { 
            background: linear-gradient(135deg, #2a2a2a, #3a3a3a);
            padding: 15px; 
            border-radius: 8px; 
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .stat-value { 
            font-size: 32px; 
            font-weight: bold; 
            color: #4CAF50; 
            margin-bottom: 5px; 
        }
        .stat-value.warning { color: #ff9800; }
        .stat-value.danger { color: #f44336; }
        .detection-details { 
            background: linear-gradient(135deg, #2a2a2a, #3a3a3a);
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .performance-info {
            background: linear-gradient(135deg, #2a2a2a, #3a3a3a);
            padding: 15px; 
            border-radius: 8px; 
            margin-top: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .method-tags {
            display: flex;
            gap: 5px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        .method-tag {
            background-color: #4CAF50;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        .method-tag.yolo { background-color: #2196F3; }
        .method-tag.color { background-color: #9C27B0; }
        .method-tag.contour { background-color: #FF9800; }
        .method-tag.edge { background-color: #F44336; }
        .controls { 
            text-align: center; 
            margin: 20px 0; 
        }
        .btn { 
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white; 
            border: none; 
            padding: 12px 24px; 
            margin: 5px; 
            border-radius: 6px; 
            cursor: pointer; 
            font-size: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        .btn:hover { 
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        }
        .confidence-bar {
            width: 100%;
            height: 25px;
            background-color: #333;
            border-radius: 12px;
            overflow: hidden;
            margin: 10px 0;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        .info-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 0;
            border-bottom: 1px solid #333;
        }
        .info-label {
            color: #888;
            font-size: 14px;
        }
        .info-value {
            font-weight: bold;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📺 TV Remote Detection System</h1>
            <p>YOLOv11 + Multi-Strategy Detection | Low-Light Optimized</p>
        </div>
        
        <div id="pickup-status" class="pickup-alert safe">
            🟢 System Active - No pickup detected
        </div>
        
        <div class="main-content">
            <div class="video-section">
                <img src="/video_feed" alt="Live Camera Feed" class="video-feed" id="video-feed">
            </div>
            
            <div class="stats-section">
                <h3>📊 Detection Status</h3>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="total-pickups">0</div>
                        <div>Total Pickups</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="actual-fps">0</div>
                        <div>Current FPS</div>
                    </div>
                </div>
                
                <div class="detection-details">
                    <h4>🎯 Current Detection</h4>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">Person:</span>
                            <span class="info-value" id="person-detected">No</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Remote:</span>
                            <span class="info-value" id="remote-detected">No</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Motion:</span>
                            <span class="info-value" id="motion">0</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Distance:</span>
                            <span class="info-value" id="distance">-</span>
                        </div>
                    </div>
                    
                    <p style="margin-top: 15px;">Pickup Confidence:</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidence-bar" style="width: 0%;">
                            <span id="confidence-text">0%</span>
                        </div>
                    </div>
                    
                    <div id="method-tags" class="method-tags"></div>
                </div>
                
                <div class="performance-info">
                    <h4>⚡ Performance</h4>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">Inference:</span>
                            <span class="info-value"><span id="inference-time">0</span>ms</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Avg FPS:</span>
                            <span class="info-value" id="fps">0</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Frames:</span>
                            <span class="info-value" id="frame-count">0</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Model:</span>
                            <span class="info-value">YOLOv11</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="location.reload()">🔄 Refresh</button>
            <button class="btn" onclick="captureFrame()">📷 Capture</button>
            <button class="btn" onclick="downloadLogs()">📥 Download Logs</button>
            <button class="btn" onclick="clearPickups()">🗑️ Clear Pickups</button>
        </div>
    </div>
    
    <script>
        let updateInterval = setInterval(updateStats, 200);
        
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    // Update counters
                    document.getElementById('total-pickups').textContent = data.remote_pickups || 0;
                    document.getElementById('actual-fps').textContent = data.actual_fps || 0;
                    document.getElementById('fps').textContent = data.fps || 0;
                    
                    // Update detection status
                    const personDetected = data.person_detected;
                    const remoteDetected = data.remote_detected;
                    
                    document.getElementById('person-detected').textContent = personDetected ? 'Yes' : 'No';
                    document.getElementById('person-detected').style.color = personDetected ? '#4CAF50' : '#888';
                    
                    document.getElementById('remote-detected').textContent = remoteDetected ? 'Yes' : 'No';
                    document.getElementById('remote-detected').style.color = remoteDetected ? '#4CAF50' : '#888';
                    
                    // Update motion and distance
                    document.getElementById('motion').textContent = Math.round(data.motion_score || 0);
                    document.getElementById('distance').textContent = 
                        data.distance_to_remote ? `${Math.round(data.distance_to_remote)}px` : '-';
                    
                    // Update confidence
                    const confidence = Math.round((data.pickup_confidence || 0) * 100);
                    document.getElementById('confidence-bar').style.width = confidence + '%';
                    document.getElementById('confidence-text').textContent = confidence + '%';
                    
                    // Color code confidence bar
                    const bar = document.getElementById('confidence-bar');
                    if (confidence > 60) {
                        bar.style.background = 'linear-gradient(90deg, #f44336, #ff6b6b)';
                    } else if (confidence > 40) {
                        bar.style.background = 'linear-gradient(90deg, #ff9800, #ffc107)';
                    } else {
                        bar.style.background = 'linear-gradient(90deg, #4CAF50, #8BC34A)';
                    }
                    
                    // Update method tags
                    const methodsDiv = document.getElementById('method-tags');
                    methodsDiv.innerHTML = '';
                    if (data.remote_detection_method) {
                        const methods = data.remote_detection_method.split(', ');
                        methods.forEach(method => {
                            const tag = document.createElement('span');
                            tag.className = `method-tag ${method}`;
                            tag.textContent = method.toUpperCase();
                            methodsDiv.appendChild(tag);
                        });
                    }
                    
                    // Update performance stats
                    document.getElementById('inference-time').textContent = Math.round(data.inference_time || 0);
                    document.getElementById('frame-count').textContent = data.frame_count || 0;
                    
                    // Update FPS color
                    const fpsElement = document.getElementById('actual-fps');
                    if (data.actual_fps < 3) {
                        fpsElement.className = 'stat-value danger';
                    } else if (data.actual_fps < 5) {
                        fpsElement.className = 'stat-value warning';
                    } else {
                        fpsElement.className = 'stat-value';
                    }
                    
                    // Update main status
                    const statusDiv = document.getElementById('pickup-status');
                    if (data.pickup_detected) {
                        statusDiv.innerHTML = '🚨 <strong>REMOTE PICKUP DETECTED!</strong>';
                        statusDiv.className = 'pickup-alert';
                    } else if (personDetected && remoteDetected) {
                        statusDiv.innerHTML = '👀 <strong>Person and Remote Detected</strong> - Monitoring...';
                        statusDiv.className = 'pickup-alert monitoring';
                    } else {
                        statusDiv.innerHTML = '🟢 <strong>System Active</strong> - No pickup detected';
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
        
        function downloadLogs() {
            alert('Log download feature would be implemented here');
        }
        
        function clearPickups() {
            if (confirm('Clear all recorded pickups?')) {
                alert('Clear pickups feature would be implemented here');
            }
        }
        
        // Initial update
        updateStats();
        
        // Reconnect video feed if connection drops
        document.getElementById('video-feed').onerror = function() {
            setTimeout(() => {
                this.src = '/video_feed?' + Date.now();
            }, 1000);
        };
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
        ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    return "No frame available", 404

def initialize_system():
    """Initialize system with YOLOv11"""
    global camera, detector
    
    try:
        logger.info("🚀 Initializing TV Remote Detection System with YOLOv11...")
        
        # Check for YOLOv11
        try:
            from ultralytics import YOLO
            logger.info("✓ Ultralytics library found")
        except ImportError:
            logger.warning("⚠️ Ultralytics not installed. Install with: pip install ultralytics")
            logger.info("Falling back to YOLOv5...")
        
        config = Config()
        camera = EnhancedLibCameraCapture(config)
        
        if not camera.start():
            logger.error("❌ Failed to start camera")
            return False
        
        detector = MultiStrategyRemoteDetector(config)
        
        logger.info("✅ System initialized successfully")
        logger.info(f"📊 Settings: {config.CAMERA_FPS} FPS, Skip {config.DETECTION_SKIP_FRAMES} frames")
        logger.info(f"🔧 Strategies enabled: YOLO + " + 
                   f"{'Color' if config.ENABLE_COLOR_DETECTION else ''} " +
                   f"{'Contour' if config.ENABLE_CONTOUR_DETECTION else ''} " +
                   f"{'Edge' if config.ENABLE_EDGE_DETECTION else ''}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False

def main():
    logger.info("📺 Starting TV Remote Detection System with YOLOv11")
    logger.info("🌙 Low-light optimizations enabled")
    
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
