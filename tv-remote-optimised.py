#!/usr/bin/env python3
"""
TV Remote Theft Detection System
Detects when someone picks up or holds the TV remote and triggers alerts
"""

import cv2
import numpy as np
import time
import logging
import subprocess
import threading
import queue
import os
import signal
from typing import Tuple, Optional, Dict, List, Set
from datetime import datetime
from collections import deque
from dataclasses import dataclass
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class TrackedObject:
    """Represents a tracked object (remote or hand)"""
    id: int
    bbox: List[int]
    center: Tuple[int, int]
    confidence: float
    last_seen: float
    positions: deque
    is_stationary: bool = True
    stationary_time: float = 0.0

@dataclass
class TheftEvent:
    """Represents a theft detection event"""
    timestamp: float
    bbox: List[int]
    confidence: float
    event_type: str  # 'pickup', 'holding', 'movement'
    frame_snapshot: Optional[np.ndarray] = None

class LibCameraCapture:
    """Optimized libcamera capture"""
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=3)
        self.process = None
        self.capture_thread = None
        self.running = False
        
    def start(self) -> bool:
        """Start camera with MJPEG"""
        try:
            self._stop()
            
            cmd = [
                'libcamera-vid',
                '--width', str(self.width),
                '--height', str(self.height),
                '--framerate', str(self.fps),
                '--timeout', '0',
                '--codec', 'mjpeg',
                '--output', '-',
                '--nopreview',
                '--flush',
                '--inline'
            ]
            
            logger.info("Starting camera...")
            
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=0,
                preexec_fn=os.setsid
            )
            
            time.sleep(2)
            
            if self.process.poll() is not None:
                return False
            
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_mjpeg_stream, daemon=True)
            self.capture_thread.start()
            
            # Wait for first frame
            start_time = time.time()
            while time.time() - start_time < 5:
                if not self.frame_queue.empty():
                    logger.info("✅ Camera ready")
                    return True
                time.sleep(0.1)
            
            return False
            
        except Exception as e:
            logger.error(f"Camera start failed: {e}")
            return False
    
    def _capture_mjpeg_stream(self):
        """Capture MJPEG frames"""
        jpeg_buffer = b""
        
        while self.running and self.process and self.process.poll() is None:
            try:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    continue
                
                jpeg_buffer += chunk
                
                while True:
                    start = jpeg_buffer.find(b'\xff\xd8')
                    if start == -1:
                        break
                        
                    end = jpeg_buffer.find(b'\xff\xd9', start + 2)
                    if end == -1:
                        break
                    
                    jpeg_data = jpeg_buffer[start:end + 2]
                    jpeg_buffer = jpeg_buffer[end + 2:]
                    
                    try:
                        nparr = np.frombuffer(jpeg_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except:
                                    pass
                            self.frame_queue.put(frame)
                            
                    except:
                        pass
                        
            except Exception:
                if self.running:
                    time.sleep(0.001)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from queue"""
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None
    
    def _stop(self):
        """Stop capture"""
        self.running = False
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=2)
            except:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass
            self.process = None
    
    def stop(self):
        self._stop()

class RemoteTheftDetector:
    """Detects TV remote theft by tracking hands and remote interactions"""
    
    def __init__(self):
        # Detection parameters
        self.min_remote_area = 800
        self.max_remote_area = 30000
        self.min_hand_area = 2000
        self.max_hand_area = 50000
        
        # Tracking
        self.tracked_remotes = {}
        self.tracked_hands = {}
        self.next_id = 0
        self.max_distance = 50  # Max distance to consider same object
        
        # Theft detection
        self.theft_events = deque(maxlen=10)
        self.last_theft_alert = 0
        self.alert_cooldown = 5.0  # Seconds between alerts
        
        # Remote detection parameters (improved)
        self.remote_colors = {
            'black': [(0, 0, 0), (180, 255, 50)],
            'gray': [(0, 0, 40), (180, 30, 150)],
            'dark': [(0, 0, 0), (180, 100, 100)]
        }
        
        # Background subtractor for motion
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=25
        )
        
        # Skin detection for hands (multiple skin tone ranges)
        self.skin_ranges = [
            # Light skin
            [(0, 20, 70), (20, 255, 255)],
            [(0, 10, 60), (20, 150, 255)],
            # Medium skin
            [(0, 20, 50), (25, 255, 255)],
            # Dark skin
            [(0, 10, 30), (20, 255, 255)],
            # Very dark skin
            [(0, 10, 0), (30, 255, 255)]
        ]
    
    def detect_theft(self, frame) -> Tuple[List[Dict], List[Dict], List[TheftEvent]]:
        """
        Main detection function
        Returns: (remote_detections, hand_detections, theft_events)
        """
        current_time = time.time()
        
        # Detect remotes
        remote_detections = self._detect_remotes(frame)
        self._update_tracked_objects(remote_detections, self.tracked_remotes, 'remote')
        
        # Detect hands
        hand_detections = self._detect_hands(frame)
        self._update_tracked_objects(hand_detections, self.tracked_hands, 'hand')
        
        # Check for theft events
        new_theft_events = self._check_theft_interactions(frame, current_time)
        
        return remote_detections, hand_detections, new_theft_events
    
    def _detect_remotes(self, frame) -> List[Dict]:
        """Improved remote detection with better filtering"""
        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for remote-like colors
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for color_name, (lower, upper) in self.remote_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphology to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_remote_area < area < self.max_remote_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Remote aspect ratio check (typically elongated)
                if 1.5 < aspect_ratio < 5.0 or 0.2 < aspect_ratio < 0.67:
                    # Additional shape analysis
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int64(box)
                    
                    # Check rectangularity
                    rect_area = rect[1][0] * rect[1][1]
                    extent = area / rect_area if rect_area > 0 else 0
                    
                    if extent > 0.65:  # Good rectangular fit
                        # Check for buttons (texture)
                        roi = frame[y:y+h, x:x+w]
                        if roi.size > 0:
                            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            texture_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                            
                            # Remotes often have button texture
                            confidence = min(0.5 + (texture_score / 1000), 0.95)
                            
                            if confidence > 0.4:  # Minimum confidence threshold
                                cx = x + w // 2
                                cy = y + h // 2
                                
                                detections.append({
                                    'bbox': [x, y, x + w, y + h],
                                    'center': (cx, cy),
                                    'confidence': confidence,
                                    'area': area,
                                    'aspect_ratio': aspect_ratio,
                                    'texture_score': texture_score
                                })
        
        return detections
    
    def _detect_hands(self, frame) -> List[Dict]:
        """Detect hands using skin color detection"""
        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        skin_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.skin_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            skin_mask = cv2.bitwise_or(skin_mask, mask)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_hand_area < area < self.max_hand_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic hand shape check
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.0:  # Hands are roughly square
                    # Check convexity defects (fingers)
                    hull = cv2.convexHull(contour, returnPoints=False)
                    if len(hull) > 3 and len(contour) > 5:
                        defects = cv2.convexityDefects(contour, hull)
                        
                        if defects is not None:
                            num_defects = len(defects)
                            confidence = min(0.5 + (num_defects * 0.1), 0.9)
                            
                            cx = x + w // 2
                            cy = y + h // 2
                            
                            detections.append({
                                'bbox': [x, y, x + w, y + h],
                                'center': (cx, cy),
                                'confidence': confidence,
                                'area': area,
                                'defects': num_defects
                            })
        
        return detections
    
    def _update_tracked_objects(self, detections: List[Dict], tracked_objects: Dict, 
                               object_type: str):
        """Update tracking for detected objects"""
        current_time = time.time()
        matched_ids = set()
        
        # Match detections to existing tracked objects
        for detection in detections:
            best_match_id = None
            best_distance = float('inf')
            
            for obj_id, tracked_obj in tracked_objects.items():
                if obj_id in matched_ids:
                    continue
                    
                # Calculate distance
                dx = detection['center'][0] - tracked_obj.center[0]
                dy = detection['center'][1] - tracked_obj.center[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match_id = obj_id
            
            if best_match_id is not None:
                # Update existing object
                tracked_obj = tracked_objects[best_match_id]
                tracked_obj.bbox = detection['bbox']
                tracked_obj.center = detection['center']
                tracked_obj.confidence = detection['confidence']
                tracked_obj.last_seen = current_time
                tracked_obj.positions.append(detection['center'])
                
                # Check if stationary
                if len(tracked_obj.positions) >= 5:
                    recent_positions = list(tracked_obj.positions)[-5:]
                    max_movement = max(
                        math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                        for p1, p2 in zip(recent_positions[:-1], recent_positions[1:])
                    )
                    
                    if max_movement < 10:
                        if not tracked_obj.is_stationary:
                            tracked_obj.is_stationary = True
                            tracked_obj.stationary_time = current_time
                    else:
                        tracked_obj.is_stationary = False
                        tracked_obj.stationary_time = 0
                
                matched_ids.add(best_match_id)
            else:
                # Create new tracked object
                new_id = self.next_id
                self.next_id += 1
                
                tracked_objects[new_id] = TrackedObject(
                    id=new_id,
                    bbox=detection['bbox'],
                    center=detection['center'],
                    confidence=detection['confidence'],
                    last_seen=current_time,
                    positions=deque([detection['center']], maxlen=30),
                    is_stationary=False,
                    stationary_time=0
                )
        
        # Remove old tracked objects
        to_remove = []
        for obj_id, tracked_obj in tracked_objects.items():
            if current_time - tracked_obj.last_seen > 2.0:  # Not seen for 2 seconds
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del tracked_objects[obj_id]
    
    def _check_theft_interactions(self, frame, current_time: float) -> List[TheftEvent]:
        """Check for hand-remote interactions indicating theft"""
        new_events = []
        
        for hand_id, hand in self.tracked_hands.items():
            for remote_id, remote in self.tracked_remotes.items():
                # Check if hand and remote overlap
                hx1, hy1, hx2, hy2 = hand.bbox
                rx1, ry1, rx2, ry2 = remote.bbox
                
                # Calculate intersection
                ix1 = max(hx1, rx1)
                iy1 = max(hy1, ry1)
                ix2 = min(hx2, rx2)
                iy2 = min(hy2, ry2)
                
                if ix2 > ix1 and iy2 > iy1:
                    # Hand and remote are overlapping
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    remote_area = (rx2 - rx1) * (ry2 - ry1)
                    overlap_ratio = intersection_area / remote_area if remote_area > 0 else 0
                    
                    if overlap_ratio > 0.3:  # Significant overlap
                        # Determine event type
                        if remote.is_stationary and not hand.is_stationary:
                            event_type = 'pickup'
                            confidence = 0.9
                        elif not remote.is_stationary and not hand.is_stationary:
                            event_type = 'holding'
                            confidence = 0.8
                        else:
                            event_type = 'movement'
                            confidence = 0.7
                        
                        # Check cooldown
                        if current_time - self.last_theft_alert > self.alert_cooldown:
                            # Create theft event
                            event = TheftEvent(
                                timestamp=current_time,
                                bbox=[ix1, iy1, ix2, iy2],
                                confidence=confidence * overlap_ratio,
                                event_type=event_type,
                                frame_snapshot=frame.copy()
                            )
                            
                            new_events.append(event)
                            self.theft_events.append(event)
                            self.last_theft_alert = current_time
                            
                            logger.warning(f"🚨 THEFT DETECTED: {event_type.upper()} - Confidence: {event.confidence:.2f}")
        
        return new_events

class TheftAlertUI:
    """UI with theft alert notifications"""
    
    def __init__(self):
        self.window_name = "TV Remote Theft Detection"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.alert_display_time = 5.0  # Seconds to show alert
        self.current_alert = None
        self.alert_start_time = 0
        
    def draw_frame(self, frame, remotes, hands, theft_events, fps) -> np.ndarray:
        """Draw detection overlay with theft alerts"""
        display = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw remote detections
        for remote in remotes:
            x1, y1, x2, y2 = remote['bbox']
            color = (0, 255, 0)  # Green for remotes
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"Remote {remote['confidence']:.2f}", 
                       (x1, y1 - 5), self.font, 0.5, color, 1)
        
        # Draw hand detections
        for hand in hands:
            x1, y1, x2, y2 = hand['bbox']
            color = (255, 0, 0)  # Blue for hands
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, "Hand", (x1, y1 - 5), self.font, 0.5, color, 1)
        
        # Update alert
        if theft_events:
            self.current_alert = theft_events[-1]
            self.alert_start_time = time.time()
        
        # Draw theft alert notification
        if self.current_alert and (time.time() - self.alert_start_time < self.alert_display_time):
            self._draw_alert(display, self.current_alert)
        
        # Status bar
        status_text = f"Remotes: {len(remotes)} | Hands: {len(hands)} | FPS: {fps:.1f}"
        cv2.rectangle(display, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(display, status_text, (10, h - 10), 
                   self.font, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(display, "Q=Quit | S=Save | SPACE=Pause", 
                   (w - 250, h - 10), self.font, 0.4, (200, 200, 200), 1)
        
        return display
    
    def _draw_alert(self, frame, theft_event: TheftEvent):
        """Draw theft alert notification"""
        h, w = frame.shape[:2]
        
        # Alert box dimensions
        alert_height = 100
        alert_width = 400
        alert_x = (w - alert_width) // 2
        alert_y = 50
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (alert_x, alert_y), 
                     (alert_x + alert_width, alert_y + alert_height), 
                     (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw alert text
        cv2.putText(frame, "!!! THEFT ALERT !!!", 
                   (alert_x + 70, alert_y + 35), 
                   self.font, 1.2, (255, 255, 255), 2)
        
        event_text = f"Remote {theft_event.event_type.upper()}"
        cv2.putText(frame, event_text, 
                   (alert_x + 100, alert_y + 65), 
                   self.font, 0.8, (255, 255, 255), 1)
        
        time_text = datetime.fromtimestamp(theft_event.timestamp).strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {time_text}", 
                   (alert_x + 120, alert_y + 85), 
                   self.font, 0.5, (200, 200, 200), 1)
        
        # Draw flashing border
        if int(time.time() * 2) % 2:  # Blink effect
            cv2.rectangle(frame, (alert_x - 2, alert_y - 2), 
                         (alert_x + alert_width + 2, alert_y + alert_height + 2), 
                         (0, 0, 255), 3)

def main():
    """Main application for TV Remote Theft Detection"""
    logger.info("🚀 Starting TV Remote Theft Detection System")
    logger.info("🔒 Monitoring for unauthorized remote access...")
    
    # Check libcamera
    try:
        result = subprocess.run(['libcamera-vid', '--version'], 
                               capture_output=True, timeout=2)
        if result.returncode != 0:
            logger.error("libcamera-vid not available")
            return
    except:
        logger.error("libcamera-vid check failed")
        return
    
    # Initialize
    camera = LibCameraCapture(640, 480, 30)
    detector = RemoteTheftDetector()
    ui = TheftAlertUI()
    
    # Start camera
    if not camera.start():
        logger.error("Failed to start camera")
        return
    
    # Create window
    cv2.namedWindow(ui.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ui.window_name, 800, 600)
    
    logger.info("✅ System armed and ready!")
    logger.info("📺 Place your remote in view. System will alert on unauthorized pickup.")
    
    # Performance tracking
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0.0
    paused = False
    
    # Theft log
    theft_log = []
    
    try:
        while True:
            if not paused:
                ret, frame = camera.read()
                if not ret or frame is None:
                    time.sleep(0.001)
                    continue
                
                # Detect theft
                remotes, hands, theft_events = detector.detect_theft(frame)
                
                # Log theft events
                for event in theft_events:
                    log_entry = {
                        'time': datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                        'type': event.event_type,
                        'confidence': event.confidence
                    }
                    theft_log.append(log_entry)
                    
                    # Save snapshot
                    filename = f"theft_{event.event_type}_{int(event.timestamp)}.jpg"
                    cv2.imwrite(filename, event.frame_snapshot)
                    logger.info(f"📸 Theft evidence saved: {filename}")
                
                # Update FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_time >= 1.0:
                    current_fps = fps_counter / (current_time - fps_time)
                    fps_counter = 0
                    fps_time = current_time
                
                # Draw UI
                display_frame = ui.draw_frame(frame, remotes, hands, theft_events, current_fps)
            else:
                cv2.putText(display_frame, "PAUSED", (10, 100), 
                           ui.font, 1.0, (0, 0, 255), 2)
            
            cv2.imshow(ui.window_name, display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, display_frame)
                logger.info(f"📸 Screenshot saved: {filename}")
            elif key == ord(' '):
                paused = not paused
                logger.info(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('l'):
                # Show theft log
                logger.info("=== THEFT LOG ===")
                for entry in theft_log[-10:]:  # Last 10 entries
                    logger.info(f"{entry['time']} - {entry['type'].upper()} (confidence: {entry['confidence']:.2f})")
    
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        logger.info(f"✅ System stopped. Total theft events detected: {len(theft_log)}")

if __name__ == "__main__":
    main()
