#!/usr/bin/env python3
import os
import cv2
import torch
import logging
import subprocess
import threading
import queue
import signal
import sys
import traceback
import psutil
import gc
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import warnings
import numpy as np
import time

# ------------------------------
# CONFIGURATION & LOGGING
# ------------------------------

warnings.filterwarnings('ignore')

class Config:
    OUTPUT_DIR: str = "output_clips"
    FRAME_SKIP: int = 3  # Process every 3rd frame
    DETECTION_CLASSES: List[str] = ['person', 'car', 'truck', 'bus']
    CLIP_DURATION_SECONDS: int = 10
    FPS: int = 10
    YOLO_MODEL: str = 'yolov5n'
    CONFIDENCE_THRESHOLD: float = 0.4
    PROCESSING_RESOLUTION: Tuple[int, int] = (320, 320)
    
    # Camera settings optimized for IMX708
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 15
    
    # Recovery settings
    MAX_CONSECUTIVE_FAILURES: int = 15
    RECOVERY_DELAY_SECONDS: int = 3
    MEMORY_CLEANUP_INTERVAL: int = 300

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/cctv_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

# ------------------------------
# WORKING CAMERA INTERFACE
# ------------------------------

class LibCameraCapture:
    def __init__(self, width=640, height=480, fps=15):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=5)
        self.process = None
        self.capture_thread = None
        self.running = False
        self.consecutive_failures = 0
        self.last_frame_time = time.time()
        
    def start(self) -> bool:
        """Start libcamera-vid capture with multiple format attempts"""
        try:
            self._stop()
            
            # Try different libcamera-vid configurations for IMX708
            configs = [
                # Configuration 1: Standard YUV420
                {
                    'codec': 'yuv420',
                    'width': self.width,
                    'height': self.height,
                    'fps': self.fps
                },
                # Configuration 2: Lower resolution for stability
                {
                    'codec': 'yuv420', 
                    'width': 320,
                    'height': 240,
                    'fps': 10
                },
                # Configuration 3: Different codec
                {
                    'codec': 'rgb',
                    'width': self.width,
                    'height': self.height,
                    'fps': self.fps
                }
            ]
            
            for i, config in enumerate(configs):
                logger.info(f"Trying libcamera-vid configuration {i+1}: {config}")
                
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
                
                # Add IMX708 specific options
                if config['codec'] == 'yuv420':
                    cmd.extend(['--inline', '--listen'])
                
                logger.info(f"Command: {' '.join(cmd)}")
                
                try:
                    self.process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=0,
                        preexec_fn=os.setsid
                    )
                    
                    # Wait for process to start
                    time.sleep(2)
                    
                    if self.process.poll() is not None:
                        stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                        logger.warning(f"Config {i+1} failed: {stderr_output}")
                        continue
                    
                    # Update dimensions for this config
                    self.width = config['width']
                    self.height = config['height']
                    self.fps = config['fps']
                    self.codec = config['codec']
                    
                    self.running = True
                    self.capture_thread = threading.Thread(target=self._capture_frames_adaptive, daemon=True)
                    self.capture_thread.start()
                    
                    # Test frame capture
                    logger.info("Testing frame capture...")
                    start_time = time.time()
                    frame_received = False
                    
                    while time.time() - start_time < 8:  # Wait up to 8 seconds
                        if not self.frame_queue.empty():
                            test_frame = self.frame_queue.get()
                            logger.info(f"✓ Configuration {i+1} works! Frame shape: {test_frame.shape}")
                            # Put frame back
                            if not self.frame_queue.full():
                                self.frame_queue.put(test_frame)
                            frame_received = True
                            break
                        time.sleep(0.2)
                    
                    if frame_received:
                        return True
                    else:
                        logger.warning(f"Configuration {i+1}: No frames received")
                        self._stop()
                        
                except Exception as e:
                    logger.error(f"Configuration {i+1} failed: {e}")
                    self._stop()
            
            logger.error("All libcamera-vid configurations failed")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            self._stop()
            return False
    
    def _capture_frames_adaptive(self):
        """Adaptive frame capture that handles different codecs"""
        if hasattr(self, 'codec') and self.codec == 'rgb':
            self._capture_rgb_frames()
        else:
            self._capture_yuv_frames()
    
    def _capture_rgb_frames(self):
        """Capture RGB frames"""
        frame_size = self.width * self.height * 3  # RGB format
        consecutive_errors = 0
        bytes_buffer = b""
        
        logger.info(f"RGB frame capture started, frame size: {frame_size}")
        
        while self.running and self.process and self.process.poll() is None:
            try:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    consecutive_errors += 1
                    if consecutive_errors > 100:
                        break
                    time.sleep(0.001)
                    continue
                
                bytes_buffer += chunk
                
                while len(bytes_buffer) >= frame_size:
                    frame_data = bytes_buffer[:frame_size]
                    bytes_buffer = bytes_buffer[frame_size:]
                    
                    try:
                        # Convert RGB to BGR
                        rgb_data = np.frombuffer(frame_data, dtype=np.uint8)
                        rgb_frame = rgb_data.reshape((self.height, self.width, 3))
                        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                        
                        if bgr_frame.shape == (self.height, self.width, 3):
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            
                            self.frame_queue.put(bgr_frame, block=False)
                            self.last_frame_time = time.time()
                            consecutive_errors = 0
                        
                    except Exception as e:
                        consecutive_errors += 1
                        if consecutive_errors % 10 == 0:
                            logger.error(f"RGB conversion error #{consecutive_errors}: {e}")
                
                if len(bytes_buffer) > frame_size * 5:
                    bytes_buffer = b""
                    
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors > 200:
                    break
                time.sleep(0.001)
    
    def _capture_yuv_frames(self):
        """Capture YUV420 frames with multiple format attempts"""
        frame_size = self.width * self.height * 3 // 2
        consecutive_errors = 0
        bytes_buffer = b""
        
        logger.info(f"YUV frame capture started, frame size: {frame_size}")
        
        while self.running and self.process and self.process.poll() is None:
            try:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    consecutive_errors += 1
                    if consecutive_errors > 100:
                        break
                    time.sleep(0.001)
                    continue
                
                bytes_buffer += chunk
                
                while len(bytes_buffer) >= frame_size:
                    frame_data = bytes_buffer[:frame_size]
                    bytes_buffer = bytes_buffer[frame_size:]
                    
                    try:
                        yuv_data = np.frombuffer(frame_data, dtype=np.uint8)
                        bgr_frame = None
                        
                        # Try different YUV conversion methods
                        conversion_methods = [
                            ('I420', cv2.COLOR_YUV2BGR_I420),
                            ('NV12', cv2.COLOR_YUV2BGR_NV12),
                            ('NV21', cv2.COLOR_YUV2BGR_NV21),
                            ('IYUV', cv2.COLOR_YUV2BGR_IYUV)
                        ]
                        
                        for method_name, color_code in conversion_methods:
                            try:
                                yuv_frame = yuv_data.reshape((self.height * 3 // 2, self.width))
                                bgr_frame = cv2.cvtColor(yuv_frame, color_code)
                                
                                if bgr_frame is not None and bgr_frame.shape == (self.height, self.width, 3):
                                    break
                                    
                            except Exception:
                                continue
                        
                        # If standard methods fail, try manual YUV conversion
                        if bgr_frame is None or bgr_frame.shape != (self.height, self.width, 3):
                            try:
                                y_size = self.width * self.height
                                u_size = v_size = y_size // 4
                                
                                y_plane = yuv_data[:y_size].reshape(self.height, self.width)
                                u_plane = yuv_data[y_size:y_size + u_size].reshape(self.height // 2, self.width // 2)
                                v_plane = yuv_data[y_size + u_size:].reshape(self.height // 2, self.width // 2)
                                
                                # Manual YUV to RGB conversion
                                u_upsampled = cv2.resize(u_plane, (self.width, self.height))
                                v_upsampled = cv2.resize(v_plane, (self.width, self.height))
                                
                                y = y_plane.astype(np.float32)
                                u = u_upsampled.astype(np.float32) - 128
                                v = v_upsampled.astype(np.float32) - 128
                                
                                r = y + 1.402 * v
                                g = y - 0.344136 * u - 0.714136 * v
                                b = y + 1.772 * u
                                
                                r = np.clip(r, 0, 255).astype(np.uint8)
                                g = np.clip(g, 0, 255).astype(np.uint8)
                                b = np.clip(b, 0, 255).astype(np.uint8)
                                
                                bgr_frame = np.stack([b, g, r], axis=2)
                                
                            except Exception as e:
                                logger.error(f"Manual YUV conversion failed: {e}")
                                continue
                        
                        if bgr_frame is not None and bgr_frame.shape == (self.height, self.width, 3):
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            
                            self.frame_queue.put(bgr_frame, block=False)
                            self.last_frame_time = time.time()
                            consecutive_errors = 0
                        else:
                            consecutive_errors += 1
                            
                    except Exception as e:
                        consecutive_errors += 1
                        if consecutive_errors % 10 == 0:
                            logger.error(f"YUV conversion error #{consecutive_errors}: {e}")
                
                if len(bytes_buffer) > frame_size * 5:
                    bytes_buffer = b""
                    
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors > 200:
                    logger.error("Too many capture errors, stopping")
                    break
                time.sleep(0.001)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame"""
        try:
            if not self.running:
                return False, None
            
            # Check if we have frames
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                self.consecutive_failures = 0
                return True, frame
            
            # Check if we're getting stale
            if time.time() - self.last_frame_time > 5:
                self.consecutive_failures += 1
                logger.warning(f"No frames for 5 seconds (failures: {self.consecutive_failures})")
            
            return False, None
            
        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"Frame read error: {e}")
            return False, None
    
    def is_healthy(self) -> bool:
        """Check if camera is working"""
        if not self.running or not self.process:
            return False
        
        # Check process status
        if self.process.poll() is not None:
            return False
        
        # Check frame freshness
        time_since_frame = time.time() - self.last_frame_time
        if time_since_frame > 10:
            return False
        
        # Check failure count
        if self.consecutive_failures > 20:
            return False
        
        return True
    
    def _stop(self):
        """Stop capture process"""
        self.running = False
        
        if self.process:
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                # Wait for clean shutdown
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass
            except:
                pass
            finally:
                self.process = None
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def stop(self):
        """Stop camera capture"""
        logger.info("Stopping camera...")
        self._stop()

# ------------------------------
# THREAT DETECTOR
# ------------------------------

class ThreatDetector:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.load_attempts = 0
        self.previous_frame = None
        self.motion_threshold = 1000
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model with retry"""
        while self.load_attempts < 3:
            try:
                logger.info(f"Loading YOLO model (attempt {self.load_attempts + 1})...")
                self.model = torch.hub.load('ultralytics/yolov5', self.config.YOLO_MODEL, pretrained=True)
                self.model.conf = self.config.CONFIDENCE_THRESHOLD
                
                # Test model
                test_img = np.zeros((320, 320, 3), dtype=np.uint8)
                _ = self.model(test_img)
                
                logger.info("✓ YOLO model loaded successfully")
                return
                
            except Exception as e:
                self.load_attempts += 1
                logger.error(f"Model load failed (attempt {self.load_attempts}): {e}")
                if self.load_attempts < 3:
                    time.sleep(3)
        
        raise Exception("Failed to load YOLO model after 3 attempts")
    
    def detect_threat(self, frame) -> Tuple[bool, np.ndarray, str]:
        """Detect threats in frame"""
        if frame is None or frame.size == 0:
            return False, np.zeros((100, 100, 3), dtype=np.uint8), ""
        
        try:
            # Motion detection
            motion_detected, motion_score = self._detect_motion(frame)
            
            # Prepare frame for YOLO
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                return False, frame, ""
            
            # Resize for processing
            new_w, new_h = self.config.PROCESSING_RESOLUTION
            frame_resized = cv2.resize(frame, (new_w, new_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # YOLO detection
            results = self.model(frame_rgb)
            detections = results.pandas().xyxy[0]
            
            # Scale detections back to original size
            if not detections.empty:
                scale_x = w / new_w
                scale_y = h / new_h
                detections['xmin'] *= scale_x
                detections['xmax'] *= scale_x
                detections['ymin'] *= scale_y
                detections['ymax'] *= scale_y
            
            # Analyze threats
            annotated_frame = frame.copy()
            threat_detected, threat_type = self._analyze_threats(detections, motion_detected, motion_score)
            
            if threat_detected:
                self._draw_detections(annotated_frame, detections)
                logger.warning(f"THREAT DETECTED: {threat_type}")
                return True, annotated_frame, threat_type
            
            return False, frame, ""
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return False, frame, ""
    
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
    
    def _analyze_threats(self, detections, motion_detected, motion_score) -> Tuple[bool, str]:
        """Analyze for threats"""
        try:
            if detections.empty:
                return False, ""
            
            # Log detections for debugging
            if not detections.empty:
                det_info = detections[['name', 'confidence']].to_dict('records')
                logger.info(f"Detections: {det_info}")
            
            # Ram raid detection (high motion + vehicle)
            if motion_detected and motion_score > 2000:
                vehicles = detections[detections['name'].isin(['car', 'truck', 'bus'])]
                if not vehicles.empty:
                    return True, "ram_raid_atm"
            
            # Person detection
            people = detections[detections['name'] == 'person']
            if not people.empty:
                return True, "person_at_atm"
            
            # Vehicle detection
            vehicles = detections[detections['name'].isin(['car', 'truck', 'bus'])]
            if not vehicles.empty:
                return True, "vehicle_near_atm"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Threat analysis error: {e}")
            return False, ""
    
    def _draw_detections(self, frame, detections):
        """Draw detection boxes"""
        try:
            for _, detection in detections.iterrows():
                if detection['confidence'] > self.config.CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
                    
                    # Validate coordinates
                    h, w = frame.shape[:2]
                    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
                    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
                    
                    class_name = detection['name']
                    confidence = detection['confidence']
                    color = (0, 255, 0) if class_name == 'person' else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
        except Exception as e:
            logger.error(f"Drawing error: {e}")

# ------------------------------
# MAIN SYSTEM
# ------------------------------

class CCTVSystem:
    def __init__(self):
        self.config = Config()
        self.camera = None
        self.detector = None
        self.running = False
        self.buffer = deque(maxlen=self.config.CLIP_DURATION_SECONDS * self.config.FPS)
        self.clip_index = 0
        self.frame_count = 0
        self.consecutive_failures = 0
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize system components"""
        try:
            logger.info("Initializing CCTV system...")
            
            # Initialize camera
            self.camera = LibCameraCapture(
                self.config.CAMERA_WIDTH,
                self.config.CAMERA_HEIGHT,
                self.config.CAMERA_FPS
            )
            
            if not self.camera.start():
                logger.error("Failed to initialize camera")
                return False
            
            # Initialize detector
            self.detector = ThreatDetector(self.config)
            
            logger.info("✓ CCTV system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def run(self):
        """Main processing loop"""
        if not self.initialize():
            return False
        
        self.running = True
        start_time = time.time()
        last_stats = time.time()
        frames_processed = 0
        
        logger.info("🎥 Starting 24/7 CCTV monitoring...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    self.consecutive_failures += 1
                    
                    if self.consecutive_failures > self.config.MAX_CONSECUTIVE_FAILURES:
                        logger.warning("Too many consecutive failures, attempting recovery...")
                        if not self._recover():
                            logger.error("Recovery failed")
                            break
                    
                    time.sleep(0.1)
                    continue
                
                self.consecutive_failures = 0
                self.buffer.append(frame.copy())
                self.frame_count += 1
                
                # Skip frames for processing efficiency
                if self.frame_count % self.config.FRAME_SKIP != 0:
                    continue
                
                frames_processed += 1
                
                # Detect threats
                is_threat, annotated_frame, threat_type = self.detector.detect_threat(frame)
                
                if is_threat:
                    self.buffer[-1] = annotated_frame
                    self._save_clip(threat_type)
                
                # Log stats every 5 minutes
                current_time = time.time()
                if current_time - last_stats > 300:
                    uptime = (current_time - start_time) / 3600
                    fps = frames_processed / (current_time - start_time)
                    logger.info(f"📊 Stats - Uptime: {uptime:.1f}h, FPS: {fps:.2f}, Clips: {self.clip_index}")
                    last_stats = current_time
                
                # Health check
                if not self.camera.is_healthy():
                    logger.warning("Camera health check failed")
                    self._recover()
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            traceback.print_exc()
        
        self._cleanup()
        return True
    
    def _recover(self) -> bool:
        """Attempt recovery"""
        try:
            logger.info("Attempting system recovery...")
            
            if self.camera:
                self.camera.stop()
                time.sleep(2)
                if not self.camera.start():
                    logger.error("Camera recovery failed")
                    return False
            
            self.buffer.clear()
            self.consecutive_failures = 0
            gc.collect()
            
            logger.info("✓ Recovery successful")
            return True
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def _save_clip(self, threat_type: str):
        """Save threat clip"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"threat_{threat_type}_{timestamp}_{self.clip_index}.mp4"
            output_path = os.path.join(self.config.OUTPUT_DIR, filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path, fourcc, self.config.FPS,
                (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT)
            )
            
            frames_saved = 0
            for frame in self.buffer:
                if frame is not None:
                    writer.write(frame)
                    frames_saved += 1
            
            writer.release()
            self.clip_index += 1
            
            logger.info(f"💾 Saved clip: {filename} ({frames_saved} frames)")
            
        except Exception as e:
            logger.error(f"Clip save error: {e}")
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        if self.camera:
            self.camera.stop()
        self.buffer.clear()
        gc.collect()

# ------------------------------
# MAIN
# ------------------------------

def main():
    setup_logging()
    logger.info("🚀 Starting 24/7 CCTV Detection System")
    
    system = CCTVSystem()
    try:
        return system.run()
    except Exception as e:
        logger.error(f"System failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    exit_code = 0 if main() else 1
    sys.exit(exit_code)
