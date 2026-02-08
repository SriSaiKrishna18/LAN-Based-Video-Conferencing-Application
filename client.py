"""
LAN Collab Client - Professional Video Conferencing
Multi-user video/audio conferencing, chat, file sharing, and screen sharing over LAN.

Features:
- Multi-user video/audio conferencing with UDP/RTP
- Screen sharing via TCP
- Group and private chat
- File sharing with upload/download
- Raise hand and emoji reactions
- Meeting recording (MP4)
- Virtual backgrounds (blur + custom images)
- Noise suppression
- Keyboard shortcuts
- Settings panel with device selection
"""

import sys
import logging
import socket
import threading
import json
import struct
import time
import random
import os
import base64
import uuid
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple
from datetime import datetime, timedelta
from fractions import Fraction
from collections import deque
import math
import wave
import tempfile

import yaml
import numpy as np
import cv2
import pyaudio
import av
from mss import mss

try:
    from scipy import signal as scipy_signal
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - noise suppression disabled")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QInputDialog, QFrame,
    QScrollArea, QListWidget, QListWidgetItem, QTabWidget, QComboBox, QMenu,
    QSlider, QCheckBox, QDialog, QDialogButtonBox, QGroupBox, QSpinBox,
    QProgressBar, QToolTip, QMessageBox, QSplitter, QStackedWidget
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QTimer, pyqtSlot, QPoint, QMutex, QMutexLocker,
    QPropertyAnimation, QEasingCurve, QRect, QSize
)
from PyQt6.QtGui import (
    QImage, QPixmap, QCursor, QKeySequence, QShortcut, QFont, QColor,
    QPainter, QBrush, QPen, QLinearGradient, QIcon
)
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# COMMON UTILITIES
# ============================================================================

def pack_rtp(payload: bytes, ssrc: int, seq: int, timestamp: int, payload_type: int, marker: bool) -> bytes:
    """Pack RTP packet."""
    version = 2
    padding = 0
    extension = 0
    csrc_count = 0
   
    byte0 = (version << 6) | (padding << 5) | (extension << 4) | csrc_count
    byte1 = (marker << 7) | payload_type
   
    header = struct.pack('>BBHII', byte0, byte1, seq, timestamp, ssrc)
    return header + payload


def unpack_rtp(data: bytes):
    """Unpack RTP packet."""
    if len(data) < 12:
        raise ValueError("RTP packet too short")
   
    byte0, byte1, seq, timestamp, ssrc = struct.unpack('>BBHII', data[:12])
   
    marker = (byte1 >> 7) & 1
    payload_type = byte1 & 0x7F
    payload = data[12:]
   
    return payload, ssrc, seq, timestamp, payload_type, marker


def write_msg(sock: socket.socket, msg: dict) -> bool:
    """Write JSON message with length prefix."""
    try:
        data = json.dumps(msg).encode('utf-8')
        length = len(data)
        sock.sendall(struct.pack('>I', length))
        sock.sendall(data)
        return True
    except:
        return False


def read_msg(sock: socket.socket) -> Optional[dict]:
    """Read JSON message with length prefix."""
    try:
        length_data = _recv_exact(sock, 4)
        if not length_data:
            return None
        length = struct.unpack('>I', length_data)[0]
        msg_data = _recv_exact(sock, length)
        if not msg_data:
            return None
        return json.loads(msg_data.decode('utf-8'))
    except:
        return None


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Receive exactly n bytes."""
    data = bytearray()
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        except:
            return None
    return bytes(data)


# ============================================================================
# MEETING RECORDER - MP4 Recording
# ============================================================================

class MeetingRecorder:
    """Records meeting video and audio to MP4 file."""
    
    def __init__(self, config: dict, output_dir: str = "recordings"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.recording = False
        self.output_container = None
        self.video_stream = None
        self.audio_stream = None
        self.start_time = None
        self.output_path = None
        self.frame_count = 0
        self.audio_samples = []
        self.lock = threading.Lock()
        
    def start(self, meeting_name: str = None) -> str:
        """Start recording. Returns output file path."""
        if self.recording:
            return self.output_path
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = meeting_name or "meeting"
        self.output_path = str(self.output_dir / f"{name}_{timestamp}.mp4")
        
        try:
            self.output_container = av.open(self.output_path, mode='w')
            
            # Video stream setup
            self.video_stream = self.output_container.add_stream('libx264', rate=15)
            self.video_stream.width = 1280
            self.video_stream.height = 720
            self.video_stream.pix_fmt = 'yuv420p'
            self.video_stream.options = {'preset': 'ultrafast', 'crf': '23'}
            
            # Audio stream setup
            self.audio_stream = self.output_container.add_stream('aac', rate=16000)
            self.audio_stream.channels = 1
            
            self.recording = True
            self.start_time = time.time()
            self.frame_count = 0
            logger.info(f"Started recording to {self.output_path}")
            return self.output_path
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return None
            
    def stop(self) -> str:
        """Stop recording. Returns output file path."""
        if not self.recording:
            return None
            
        self.recording = False
        
        try:
            with self.lock:
                if self.output_container:
                    # Flush encoder
                    for packet in self.video_stream.encode(None):
                        self.output_container.mux(packet)
                    self.output_container.close()
                    self.output_container = None
                    
            logger.info(f"Recording saved to {self.output_path}")
            return self.output_path
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return None
            
    def add_frame(self, frame: np.ndarray):
        """Add video frame to recording."""
        if not self.recording or frame is None:
            return
            
        try:
            with self.lock:
                if not self.output_container:
                    return
                    
                # Resize if needed
                h, w = frame.shape[:2]
                if w != 1280 or h != 720:
                    frame = cv2.resize(frame, (1280, 720))
                
                # Convert to YUV
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                av_frame = av.VideoFrame.from_ndarray(yuv, format='yuv420p')
                av_frame.pts = self.frame_count
                
                for packet in self.video_stream.encode(av_frame):
                    self.output_container.mux(packet)
                    
                self.frame_count += 1
        except Exception as e:
            logger.debug(f"Recording frame error: {e}")
            
    def add_audio(self, audio_data: bytes):
        """Add audio data to recording."""
        if not self.recording or not audio_data:
            return
            
        try:
            with self.lock:
                if not self.output_container:
                    return
                    
                # Convert bytes to numpy array
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_np = audio_np.astype(np.float32) / 32768.0
                
                # Create audio frame
                audio_frame = av.AudioFrame.from_ndarray(
                    audio_np.reshape(1, -1), format='fltp', layout='mono'
                )
                audio_frame.sample_rate = 16000
                
                for packet in self.audio_stream.encode(audio_frame):
                    self.output_container.mux(packet)
        except Exception as e:
            logger.debug(f"Recording audio error: {e}")


# ============================================================================
# VIRTUAL BACKGROUND - Blur & Custom Images
# ============================================================================

class VirtualBackgroundProcessor:
    """Applies virtual backgrounds: blur or custom images."""
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = False
        self.mode = "none"  # none, blur, image
        self.blur_strength = config.get("virtual_background", {}).get("blur_strength", 21)
        self.background_image = None
        self.custom_backgrounds_dir = Path(
            config.get("virtual_background", {}).get("custom_backgrounds_dir", "backgrounds")
        )
        self.custom_backgrounds_dir.mkdir(exist_ok=True)
        
        # Simple background subtraction for person detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        
    def set_mode(self, mode: str, image_path: str = None):
        """Set virtual background mode."""
        self.mode = mode
        if mode == "image" and image_path:
            self.background_image = cv2.imread(image_path)
            if self.background_image is not None:
                self.enabled = True
            else:
                logger.warning(f"Could not load background image: {image_path}")
        elif mode == "blur":
            self.enabled = True
        else:
            self.enabled = False
            
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply virtual background to frame."""
        if not self.enabled or frame is None:
            return frame
            
        try:
            if self.mode == "blur":
                return self._apply_blur(frame)
            elif self.mode == "image":
                return self._apply_custom_background(frame)
            return frame
        except Exception as e:
            logger.debug(f"Virtual background error: {e}")
            return frame
            
    def _apply_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply background blur effect."""
        # Create person mask using background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.GaussianBlur(fg_mask, (15, 15), 0)
        
        # Normalize mask
        fg_mask = fg_mask.astype(np.float32) / 255.0
        fg_mask = np.stack([fg_mask] * 3, axis=-1)
        
        # Create blurred background
        blurred = cv2.GaussianBlur(frame, (self.blur_strength, self.blur_strength), 0)
        
        # Blend
        result = (frame * fg_mask + blurred * (1 - fg_mask)).astype(np.uint8)
        return result
        
    def _apply_custom_background(self, frame: np.ndarray) -> np.ndarray:
        """Apply custom background image."""
        if self.background_image is None:
            return frame
            
        # Resize background to match frame
        h, w = frame.shape[:2]
        bg = cv2.resize(self.background_image, (w, h))
        
        # Create person mask
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.GaussianBlur(fg_mask, (15, 15), 0)
        fg_mask = fg_mask.astype(np.float32) / 255.0
        fg_mask = np.stack([fg_mask] * 3, axis=-1)
        
        # Blend person over background
        result = (frame * fg_mask + bg * (1 - fg_mask)).astype(np.uint8)
        return result
        
    def get_available_backgrounds(self) -> List[str]:
        """Get list of available custom backgrounds."""
        if not self.custom_backgrounds_dir.exists():
            return []
        return [str(p) for p in self.custom_backgrounds_dir.glob("*.jpg")] + \
               [str(p) for p in self.custom_backgrounds_dir.glob("*.png")]


# ============================================================================
# NOISE SUPPRESSION
# ============================================================================

class NoiseSuppressor:
    """Simple noise suppression using high-pass filter."""
    
    def __init__(self, sample_rate: int = 16000, enabled: bool = True):
        self.sample_rate = sample_rate
        self.enabled = enabled
        self._filter_coeffs = None
        
        if SCIPY_AVAILABLE:
            # High-pass filter to remove low-frequency noise
            nyquist = sample_rate / 2
            cutoff = 100 / nyquist  # 100 Hz cutoff
            self._filter_coeffs = scipy_signal.butter(4, cutoff, btype='high')
        
    def process(self, audio_data: bytes) -> bytes:
        """Apply noise suppression to audio data."""
        if not self.enabled or not SCIPY_AVAILABLE or audio_data is None:
            return audio_data
            
        try:
            # Convert to numpy
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Apply high-pass filter
            b, a = self._filter_coeffs
            filtered = scipy_signal.filtfilt(b, a, audio_np)
            
            # Clip and convert back
            filtered = np.clip(filtered, -32768, 32767).astype(np.int16)
            return filtered.tobytes()
        except Exception as e:
            logger.debug(f"Noise suppression error: {e}")
            return audio_data


# ============================================================================
# REACTIONS MANAGER
# ============================================================================

class ReactionManager:
    """Manages emoji reactions with animations."""
    
    REACTIONS = ["👍", "❤️", "👏", "😂", "😮", "🎉", "🔥", "💯"]
    
    def __init__(self):
        self.active_reactions: List[Dict] = []
        self.lock = threading.Lock()
        
    def add_reaction(self, emoji: str, username: str):
        """Add a reaction to display."""
        with self.lock:
            self.active_reactions.append({
                "emoji": emoji,
                "username": username,
                "timestamp": time.time(),
                "x": random.randint(100, 500),
                "y": 400,
                "vy": -3  # Upward velocity
            })
            
    def update(self) -> List[Dict]:
        """Update and return active reactions."""
        now = time.time()
        with self.lock:
            # Update positions
            for r in self.active_reactions:
                r["y"] += r["vy"]
                r["vy"] += 0.1  # Gravity
                
            # Remove old reactions
            self.active_reactions = [
                r for r in self.active_reactions 
                if now - r["timestamp"] < 3.0 and r["y"] < 600
            ]
            return list(self.active_reactions)


# ============================================================================
# AUDIO LEVEL MONITOR
# ============================================================================

class AudioLevelMonitor:
    """Monitors audio levels for speaking indicator."""
    
    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self.current_level = 0.0
        self.is_speaking = False
        self.history = deque(maxlen=10)
        
    def update(self, audio_data: bytes) -> float:
        """Update with audio data, return level 0-1."""
        if not audio_data:
            return 0.0
            
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(audio_np ** 2)) / 32768.0
            self.history.append(rms)
            self.current_level = np.mean(self.history)
            self.is_speaking = self.current_level > self.threshold
            return min(1.0, self.current_level * 10)
        except:
            return 0.0


# ============================================================================
# CONTROL CLIENT
# ============================================================================

class ControlClient:
    """TCP-based control client."""
   
    def __init__(self, config: dict, gui_callback: Optional[Callable] = None):
        self.config = config
        self.gui_callback = gui_callback
        self.server_host = config["server"]["host"]
        self.server_port = config["server"]["control_tcp_port"]
        self.socket = None
        self.connected = False
        self.user_id = None
        self.username = None
        self.receive_thread = None
        self.running = False
   
    def connect(self, username: str) -> bool:
        """Connect to the control server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
           
            join_msg = {"type": "join", "username": username}
            self._send_message(join_msg)
           
            response = self._receive_message()
            if not response or response.get("type") != "join_ack":
                return False
           
            self.user_id = response.get("user_id")
            self.username = username
            self.connected = True
            self.running = True
           
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
           
            logger.info(f"Connected as {username}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
   
    def disconnect(self):
        """Disconnect from server."""
        self.running = False
        self.connected = False
        if self.socket:
            try:
                self._send_message({"type": "disconnect"})
                self.socket.close()
            except:
                pass
   
    def send_chat(self, message: str, to: Optional[str] = None):
        """Send chat message."""
        if not self.connected:
            return
        msg = {"type": "chat", "text": message}
        if to:
            msg["to"] = to
        self._send_message(msg)
   
    def _send_message(self, msg: dict):
        """Send JSON message with length prefix."""
        data = json.dumps(msg).encode('utf-8')
        self.socket.sendall(struct.pack('>I', len(data)))
        self.socket.sendall(data)
   
    def _receive_message(self) -> Optional[dict]:
        """Receive JSON message."""
        try:
            length_data = _recv_exact(self.socket, 4)
            if not length_data:
                return None
            length = struct.unpack('>I', length_data)[0]
            msg_data = _recv_exact(self.socket, length)
            if not msg_data:
                return None
            return json.loads(msg_data.decode('utf-8'))
        except:
            return None
   
    def _receive_loop(self):
        """Receive messages from server."""
        while self.running:
            try:
                msg = self._receive_message()
                if not msg:
                    break
                self._handle_message(msg)
            except:
                break
   
    def _handle_message(self, msg: dict):
        """Handle message from server."""
        if self.gui_callback:
            msg_type = msg.get("type")
            if msg_type == "chat":
                self.gui_callback("chat", msg)
            elif msg_type == "roster":
                self.gui_callback("roster", msg)
            elif msg_type == "file_available":
                self.gui_callback("file_available", msg)
            elif msg_type == "reaction":
                self.gui_callback("reaction", msg)
            elif msg_type == "hand_raise":
                self.gui_callback("hand_raise", msg)
    
    def send_message(self, msg: dict):
        """Send arbitrary message to server (for new feature types)."""
        if self.connected:
            self._send_message(msg)


# ============================================================================
# FILE CLIENT
# ============================================================================

class FileClient:
    """TCP-based file client."""
   
    def __init__(self, config: dict):
        self.config = config
        self.server_host = config["server"]["host"]
        self.server_port = config["server"]["file_tcp_port"]
   
    def upload_file(self, filepath: str, progress_callback=None) -> str:
        """Upload file to server."""
        try:
            path = Path(filepath)
            if not path.exists():
                return None
           
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(60.0)
            sock.connect((self.server_host, self.server_port))
           
            request = {
                "action": "upload",
                "filename": path.name,
                "size": path.stat().st_size
            }
            sock.sendall((json.dumps(request) + "\n").encode())
           
            response_data = b""
            while b"\n" not in response_data:
                chunk = sock.recv(4096)
                if not chunk:
                    return None
                response_data += chunk
           
            response = json.loads(response_data.split(b"\n")[0].decode())
            if response.get("status") != "ok":
                return None
           
            file_id = response.get("file_id")
           
            with open(filepath, 'rb') as f:
                file_data = f.read()
           
            sock.sendall(file_data)
            sock.close()
           
            logger.info(f"Uploaded {path.name}")
            return file_id
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None
   
    def download_file(self, file_id: str, save_path: str, progress_callback=None) -> bool:
        """Download file from server."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(60.0)
            sock.connect((self.server_host, self.server_port))
           
            request = {"action": "download", "file_id": file_id}
            sock.sendall((json.dumps(request) + "\n").encode())
           
            response_data = b""
            while b"\n" not in response_data:
                chunk = sock.recv(4096)
                if not chunk:
                    return False
                response_data += chunk
           
            response_line, remaining = response_data.split(b"\n", 1)
            response = json.loads(response_line.decode())
           
            if response.get("status") != "ok":
                return False
           
            filesize = response.get("size")
            received = len(remaining)
            file_data = bytearray(remaining)
           
            while received < filesize:
                chunk = sock.recv(min(65536, filesize - received))
                if not chunk:
                    return False
                file_data.extend(chunk)
                received += len(chunk)
           
            Path(save_path).write_bytes(file_data)
            sock.close()
           
            logger.info(f"Downloaded to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
   
    def list_files(self):
        """Get list of available files."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((self.server_host, self.server_port))
           
            request = {"action": "list"}
            sock.sendall((json.dumps(request) + "\n").encode())
           
            response_data = b""
            while b"\n" not in response_data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
           
            response = json.loads(response_data.split(b"\n")[0].decode())
            sock.close()
           
            return response.get("files", [])
        except:
            return []


# ============================================================================
# AUDIO SENDER/RECEIVER
# ============================================================================

class AudioSender:
    """UDP-based audio sender."""
   
    def __init__(self, config: dict, shared_socket=None):
        self.config = config
        self.server_host = config["server"]["host"]
        self.server_port = config["media"]["audio_rtp_port"]
        self.socket = shared_socket
        self.local_port = int(config["media"]["local_audio_port"])
        self.audio = None
        self.stream = None
        self.running = False
        self.send_thread = None
        self.ssrc = None
        self.seq = 0
        self.timestamp = 0
   
    def start(self):
        """Start audio capture."""
        try:
            if self.socket is None:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.bind(('0.0.0.0', self.local_port))
           
            unique_seed = f"{time.time()}{random.random()}{os.getpid()}"
            self.ssrc = hash(unique_seed) & 0xFFFFFFFF
           
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=960
            )
           
            self.running = True
            self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self.send_thread.start()
            logger.info("Audio sender started")
            return True
        except Exception as e:
            logger.error(f"Failed to start audio sender: {e}")
            return False
   
    def stop(self):
        """Stop audio capture."""
        self.running = False
       
        # Wait for send thread
        if self.send_thread and self.send_thread.is_alive():
            self.send_thread.join(timeout=1.0)
       
        # Stop and close stream
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            finally:
                self.stream = None
       
        # Terminate PyAudio
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            finally:
                self.audio = None
       
        logger.info("Audio sender stopped")
   
    def _send_loop(self):
        """Capture and send audio."""
        while self.running:
            try:
                audio_data = self.stream.read(960, exception_on_overflow=False)
                rtp_packet = pack_rtp(audio_data, self.ssrc, self.seq, self.timestamp, 0, False)
                self.socket.sendto(rtp_packet, (self.server_host, self.server_port))
                self.seq = (self.seq + 1) & 0xFFFF
                self.timestamp = (self.timestamp + 960) & 0xFFFFFFFF
                time.sleep(0.02)
            except:
                pass


class AudioReceiver:
    """UDP-based audio receiver."""
   
    def __init__(self, config: dict):
        self.config = config
        self.server_host = config["server"]["host"]
        self.server_port = config["media"]["audio_rtp_port"]
        self.local_port = int(config["media"]["local_audio_port"])
        self.socket = None
        self.running = False
        self.receive_thread = None
        self.audio = None
        self.stream = None
   
    def get_socket(self):
        """Expose socket for sender."""
        return self.socket
   
    def start(self):
        """Start audio receiver."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.local_port))
            self.socket.sendto(b'AUDIO_RECEIVER_REGISTER', (self.server_host, self.server_port))
           
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                output=True,
                frames_per_buffer=960
            )
           
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            logger.info("Audio receiver started")
            return True
        except Exception as e:
            logger.error(f"Failed to start audio receiver: {e}")
            return False
   
    def stop(self):
        """Stop audio receiver."""
        self.running = False
       
        # Wait for receive thread
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)
       
        # Stop and close stream
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            finally:
                self.stream = None
       
        # Terminate PyAudio
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            finally:
                self.audio = None
       
        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
       
        logger.info("Audio receiver stopped")
   
    def _receive_loop(self):
        """Receive and play audio."""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(2048)
                if len(data) < 12:
                    continue
                payload, ssrc, seq, timestamp, pt, marker = unpack_rtp(data)
                if self.stream and self.stream.is_active():
                    self.stream.write(payload, exception_on_underflow=False)
            except:
                pass


# ============================================================================
# VIDEO SENDER/RECEIVER
# ============================================================================

class VideoSender:
    """UDP-based video sender."""
   
    def __init__(self, config: dict, shared_socket=None):
        self.config = config
        self.server_host = config["server"]["host"]
        self.server_port = config["media"]["video_rtp_port"]
        self.socket = shared_socket
        self.local_port = int(config["media"]["local_video_port"])
        self.camera = None
        self.running = False
        self.send_thread = None
        self.ssrc = None
        self.seq = 0
        self.timestamp = 0
        self.codec_ctx = None
        self.last_frame = None
   
    def start(self) -> bool:
        """Start video capture."""
        try:
            if self.socket is None:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.bind(('0.0.0.0', self.local_port))
           
            device_index = self.config.get("devices", {}).get("video_device_index", 0)
            system = platform.system()
           
            if system == "Windows":
                self.camera = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            elif system == "Linux":
                self.camera = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
            else:
                self.camera = cv2.VideoCapture(device_index)
           
            if not self.camera.isOpened():
                return False
           
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 15)
           
            codec = av.codec.Codec('libx264', 'w')
            self.codec_ctx = codec.create()
            self.codec_ctx.width = 640
            self.codec_ctx.height = 480
            self.codec_ctx.pix_fmt = 'yuv420p'
            self.codec_ctx.time_base = Fraction(1, 15)
            self.codec_ctx.framerate = 15
            self.codec_ctx.bit_rate = 500000
            self.codec_ctx.options = {
                'preset': 'ultrafast',
                'tune': 'zerolatency',
                'profile': 'baseline',
            }
           
            unique_seed = f"{time.time()}{random.random()}{os.getpid()}"
            self.ssrc = hash(unique_seed) & 0xFFFFFFFF
           
            self.running = True
            self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self.send_thread.start()
            logger.info("Video sender started")
            return True
        except Exception as e:
            logger.error(f"Failed to start video sender: {e}")
            return False
   
    def stop(self):
        """Stop video capture."""
        self.running = False
       
        # Wait for send thread to finish
        if self.send_thread and self.send_thread.is_alive():
            self.send_thread.join(timeout=1.0)
       
        # Close camera
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None
       
        # Close codec
        if self.codec_ctx:
            try:
                # Flush encoder
                packets = self.codec_ctx.encode(None)
                for _ in packets:
                    pass
            except:
                pass
            self.codec_ctx = None
       
        cv2.destroyAllWindows()
        logger.info("Video sender stopped")
   
    def _send_loop(self):
        """Capture and send video."""
        frame_count = 0
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
               
                # Store frame BEFORE encoding for self-preview
                self.last_frame = frame.copy()
               
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                av_frame = av.VideoFrame.from_ndarray(yuv, format='yuv420p')
                av_frame.pts = frame_count
               
                packets = self.codec_ctx.encode(av_frame)
                for packet in packets:
                    rtp_packet = pack_rtp(bytes(packet), self.ssrc, self.seq, self.timestamp, 96, False)
                    self.socket.sendto(rtp_packet, (self.server_host, self.server_port))
                    self.seq = (self.seq + 1) & 0xFFFF
               
                self.timestamp += 3000
                frame_count += 1
                time.sleep(1.0 / 15.0)
            except Exception as e:
                logger.error(f"Video send error: {e}")
                pass


class VideoReceiver:
    """UDP-based video receiver."""
   
    def __init__(self, config: dict, frame_callback=None):
        self.config = config
        self.frame_callback = frame_callback
        self.server_host = config["server"]["host"]
        self.server_port = config["media"]["video_rtp_port"]
        self.local_port = int(config["media"]["local_video_port"])
        self.socket = None
        self.running = False
        self.thread = None
        self.decoders = {}
   
    def get_socket(self):
        """Expose socket for sender."""
        return self.socket
   
    def start(self):
        """Start video receiver."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.local_port))
            self.socket.sendto(b'VIDEO_RECEIVER_REGISTER', (self.server_host, self.server_port))
           
            self.running = True
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            logger.info("Video receiver started")
            return True
        except Exception as e:
            logger.error(f"Failed to start video receiver: {e}")
            return False
   
    def stop(self):
        """Stop video receiver."""
        self.running = False
   
    def _receive_loop(self):
        """Receive and decode video."""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536)
                if len(data) < 12:
                    continue
               
                payload, ssrc, seq, timestamp, pt, marker = unpack_rtp(data)
               
                if ssrc not in self.decoders:
                    codec = av.codec.Codec('h264', 'r')
                    self.decoders[ssrc] = codec.create()
               
                decoder = self.decoders[ssrc]
                packet = av.Packet(payload)
                frames = decoder.decode(packet)
                for frame in frames:
                    img = frame.to_ndarray(format='bgr24')
                    if self.frame_callback:
                        self.frame_callback(ssrc, img)
            except:
                pass


# ============================================================================
# SCREEN SHARING
# ============================================================================

class ScreenSharer:
    """Screen capture and TCP transmission."""
   
    def __init__(self, config: dict):
        self.config = config
        self.host = config["server"]["host"]
        self.port = config["server"]["screen_tcp_port"]
        self.socket = None
        self.running = False
        self.capture_thread = None
   
    def start(self, monitor=1, quality=50, fps=8):
        """Start screen sharing."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.connect((self.host, self.port))
           
            if not write_msg(self.socket, {"role": "presenter"}):
                return False
           
            response = read_msg(self.socket)
            if not response or response.get("status") != "ok":
                return False
           
            self.running = True
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(monitor, quality, fps),
                daemon=True
            )
            self.capture_thread.start()
            logger.info("Screen sharing started")
            return True
        except Exception as e:
            logger.error(f"Failed to start screen sharing: {e}")
            return False
   
    def stop(self):
        """Stop screen sharing."""
        self.running = False
        if self.socket:
            try:
                write_msg(self.socket, {"type": "end"})
                self.socket.close()
            except:
                pass
   
    def _capture_loop(self, monitor: int, quality: int, fps: int):
        """Capture and send screen frames."""
        frame_duration = 1.0 / fps
       
        try:
            with mss() as sct:
                monitor_config = sct.monitors[monitor]
               
                while self.running:
                    loop_start = time.time()
                   
                    screenshot = sct.grab(monitor_config)
                    frame = np.array(screenshot, dtype=np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                   
                    height, width = frame.shape[:2]
                    if width > 960:
                        scale = 960 / width
                        new_width = 960
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                        width, height = new_width, new_height
                   
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    success, encoded = cv2.imencode('.jpg', frame, encode_params)
                   
                    if success:
                        frame_b64 = base64.b64encode(encoded.tobytes()).decode('ascii')
                        if not write_msg(self.socket, {
                            "type": "frame",
                            "data": frame_b64,
                            "width": width,
                            "height": height
                        }):
                            break
                   
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, frame_duration - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except:
            pass


class ScreenViewer:
    """Receive shared screen."""
   
    def __init__(self, config: dict, frame_callback=None):
        self.config = config
        self.host = config["server"]["host"]
        self.port = config["server"]["screen_tcp_port"]
        self.frame_callback = frame_callback
        self.socket = None
        self.running = False
        self.receive_thread = None
   
    def start(self):
        """Start receiving screen frames."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.connect((self.host, self.port))
           
            if not write_msg(self.socket, {"role": "viewer"}):
                return False
           
            response = read_msg(self.socket)
            if not response or response.get("status") != "ok":
                return False
           
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            logger.info("Screen viewer started")
            return True
        except Exception as e:
            logger.error(f"Failed to start screen viewer: {e}")
            return False
   
    def stop(self):
        """Stop receiving screen frames."""
        self.running = False
        if self.frame_callback:
            self.frame_callback(None, 0, 0)
   
    def _receive_loop(self):
        """Receive screen frames."""
        try:
            while self.running:
                msg = read_msg(self.socket)
                if not msg:
                    break
               
                if msg.get("type") == "end":
                    if self.frame_callback:
                        self.frame_callback(None, 0, 0)
                    continue
               
                if msg.get("type") == "frame":
                    frame_b64 = msg["data"]
                    frame_bytes = base64.b64decode(frame_b64)
                    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                    img = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                   
                    if img is not None and self.frame_callback:
                        self.frame_callback(img, msg["width"], msg["height"])
        except:
            pass
        finally:
            if self.frame_callback:
                self.frame_callback(None, 0, 0)


# ============================================================================
# GUI
# ============================================================================

class VideoLabel(QLabel):
    """Custom QLabel for video with context menu."""
   
    def __init__(self, ssrc=None, parent_window=None, is_self=False):
        super().__init__()
        self.ssrc = ssrc
        self.parent_window = parent_window
        self.is_self = is_self
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
   
    def _show_context_menu(self, pos: QPoint):
        if not self.parent_window:
            return
       
        menu = QMenu(self)
       
        # Find username for this label
        pin_identifier = "SELF" if self.is_self else None
        if not self.is_self:
            # Find username by reverse lookup
            for username, label in self.parent_window.participant_slots.items():
                if label == self:
                    pin_identifier = username
                    break
       
        if self.parent_window.pinned_ssrc == pin_identifier:
            action = menu.addAction("📌 Unpin")
            action.triggered.connect(lambda: self.parent_window.unpin_video())
        else:
            pin_text = "📍 Pin My Video" if self.is_self else "📍 Pin This Video"
            action = menu.addAction(pin_text)
            action.triggered.connect(lambda: self.parent_window.pin_video(pin_identifier))
       
        menu.exec(self.mapToGlobal(pos))


class MainWindow(QMainWindow):
    frame_received = pyqtSignal(int, np.ndarray)
    screen_received = pyqtSignal(object, int, int)
    chat_received = pyqtSignal(str, str, bool, str)
    roster_updated = pyqtSignal(list)
    file_available = pyqtSignal(str, str, int)
    reaction_received = pyqtSignal(str, str)  # emoji, username
    hand_raised = pyqtSignal(str, bool)  # username, raised

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.setWindowTitle("LAN Video Conference")
        self.resize(1600, 900)
       
        self.video_mutex = QMutex()
        self.screen_mutex = QMutex()
       
        self.control = None
        self.file_client = None
        self.video_sender = None
        self.video_receiver = None
        self.audio_sender = None
        self.audio_receiver = None
        self.screen_sharer = None
        self.screen_viewer = None
       
        self.available_files = {}
        self.video_labels = {}
        self.stream_last_frame_time = {}
        self.screen_thumbnails = {}
        self.screen_presenter_video = None
        self.screen_presenting_ssrc = None
        self.presenter_last_frame = None
        self.pinned_ssrc = None
        self.my_ssrc = None
        self.online_users = []
        self.camera_active = False
        self.mic_active = False
        self.screen_active = False
        self.viewing_screen = False
        self.chat_visible = False
        self.files_visible = False
        self.participants_visible = False
        self.chat_containers = {}
        self.preview_running = False
        self.screen_label = None
        self.screen_thumbnail_container = None
        self.meeting_start_time = datetime.now()
       
        # New feature state
        self.hand_raised_state = False
        self.is_recording = False
        self.is_fullscreen = False
        self.raised_hands = {}  # username -> bool
       
        # New feature instances
        self.meeting_recorder = MeetingRecorder(config)
        self.virtual_bg = VirtualBackgroundProcessor(config)
        self.noise_suppressor = NoiseSuppressor()
        self.reaction_manager = ReactionManager()
        self.audio_level_monitor = AudioLevelMonitor()
       
        self.my_username = self._get_username()
        self.participant_slots = {}  # Maps username to video label
        self.username_to_ssrc = {}   # Maps username to SSRC
        self.ssrc_to_username = {}   # Maps SSRC to username (reverse lookup)
        self._build_ui()
       
        self.frame_received.connect(self._update_video_display)
        self.screen_received.connect(self._update_screen_display)
        self.chat_received.connect(self._add_chat_message_slot)
        self.roster_updated.connect(self._on_roster_slot)
        self.file_available.connect(self._on_file_available_slot)
        self.reaction_received.connect(self._on_reaction_received)
        self.hand_raised.connect(self._on_hand_raised)
       
        self._start_control()
        self._setup_file_client()
        self._setup_media()
        self._start_meeting_timer()
        self._start_stream_timeout_checker()
        self._setup_keyboard_shortcuts()

    def _get_username(self) -> str:
        default_name = f"User{random.randint(1000,9999)}"
        name, ok = QInputDialog.getText(
            self, "Enter Username", "Choose your display name:",
            QLineEdit.EchoMode.Normal, default_name
        )
        return name.strip() if ok and name.strip() else default_name

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
        """)
       
        root = QVBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)
       
        # Top bar
        top = QFrame()
        top.setFixedHeight(60)
        top.setStyleSheet("background-color: rgba(0,0,0,0.3); border-bottom: 2px solid #0078d4;")
        hb = QHBoxLayout(top)
       
        title = QLabel("🎥 LAN Collab")
        title.setStyleSheet("color:#00d4ff; font-size:24px; font-weight:bold; padding:10px;")
        hb.addWidget(title)
       
        self.timer_label = QLabel("⏱️ 00:00:00")
        self.timer_label.setStyleSheet("color:#00ff88; font-size:16px; font-weight:bold; padding:10px;")
        hb.addWidget(self.timer_label)
        hb.addStretch()
       
        self._user_lbl = QLabel(f"👤 {self.my_username}")
        self._user_lbl.setStyleSheet("color:#fff; font-size:14px; padding:10px;")
        hb.addWidget(self._user_lbl)
        root.addWidget(top)
       
        # Video container
        self.video_container = QWidget()
        self.video_layout = QVBoxLayout(self.video_container)
        self.video_layout.setContentsMargins(15,15,15,15)
        self.video_layout.setSpacing(15)
       
        # Self view
        self.self_view = VideoLabel(parent_window=self, is_self=True)
        self.self_view.setFixedSize(220,165)
        self.self_view.setStyleSheet("""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #00d4ff, stop:1 #0078d4);
            border: 3px solid #00ff88; border-radius: 12px; color: white; font-size: 14px; font-weight: bold;
        """)
        self.self_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.self_view.setText("📹 You\n(Camera Off)")
        self.self_view.setScaledContents(True)
        self.self_view.setParent(self.video_container)
       
        root.addWidget(self.video_container)
       
        # Controls
        controls = self._build_controls()
        root.addWidget(controls)
       
        # Side panels
        self.chat_panel = self._build_chat_panel()
        self.chat_panel.setParent(central)
        self.chat_panel.hide()
       
        self.files_panel = self._build_files_panel()
        self.files_panel.setParent(central)
        self.files_panel.hide()
       
        self.participants_panel = self._build_participants_panel()
        self.participants_panel.setParent(central)
        self.participants_panel.hide()

    def _build_controls(self):
        bar = QFrame()
        bar.setFixedHeight(90)
        bar.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a1a2e, stop:1 #16213e); border-top: 2px solid #0078d4;")
        hb = QHBoxLayout(bar)
        hb.setContentsMargins(30,15,30,15)
       
        self.mic_btn = self._round_btn("🎤","Mic", self.toggle_mic, "#8e44ad")
        self.cam_btn = self._round_btn("📹","Camera", self.toggle_camera, "#2980b9")
        self.scr_btn = self._round_btn("🖥️","Screen", self.toggle_screen, "#16a085")
       
        hb.addWidget(self.mic_btn)
        hb.addWidget(self.cam_btn)
        hb.addWidget(self.scr_btn)
       
        # New feature buttons
        self.hand_btn = self._round_btn("✋","Hand", self.toggle_raise_hand, "#e67e22")
        self.react_btn = self._round_btn("😀","React", self._show_reaction_menu, "#e91e63")
        self.record_btn = self._round_btn("⏺️","Record", self.toggle_recording, "#c0392b")
       
        hb.addWidget(self.hand_btn)
        hb.addWidget(self.react_btn)
        hb.addWidget(self.record_btn)
       
        hb.addStretch()
       
        self.chat_btn = self._round_btn("💬","Chat", self.toggle_chat, "#27ae60")
        self.files_btn= self._round_btn("📁","Files", self.toggle_files,"#f39c12")
        self.participants_btn = self._round_btn("👥","People", self.toggle_participants,"#9b59b6")
        self.settings_btn = self._round_btn("⚙️","Settings", self._show_settings, "#607d8b")
       
        hb.addWidget(self.chat_btn)
        hb.addWidget(self.files_btn)
        hb.addWidget(self.participants_btn)
        hb.addWidget(self.settings_btn)
        hb.addStretch()
        hb.addWidget(self._round_btn("📞","Leave", self.close, "#e74c3c"))
       
        return bar

    def _round_btn(self, emoji, text, cb, color="#3a3a3a"):
        c = QWidget()
        vb = QVBoxLayout(c)
        vb.setSpacing(8)
        vb.setContentsMargins(5,0,5,0)
       
        btn = QPushButton(emoji)
        btn.setFixedSize(60,60)
        btn.setStyleSheet(f"""
            QPushButton{{background-color:{color};color:white;border:none;border-radius:30px;font-size:24px;}}
            QPushButton:hover{{background-color:{color};border:3px solid #00ff88;}}
        """)
        btn.clicked.connect(cb)
       
        lab = QLabel(text)
        lab.setStyleSheet("color:#ddd;font-size:12px;font-weight:bold;")
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
       
        vb.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
        vb.addWidget(lab)
       
        c.button = btn
        c.default_color = color
        return c

    def _build_chat_panel(self):
        panel = QFrame()
        panel.setStyleSheet("QFrame{background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1e2749, stop:1 #1a1a2e); border:2px solid #00d4ff; border-radius:15px;}")
        v = QVBoxLayout(panel)
        v.setContentsMargins(15,15,15,15)
       
        hdr = QWidget()
        hb = QHBoxLayout(hdr)
        hb.setContentsMargins(0,0,0,10)
       
        t = QLabel("💬 Chat")
        t.setStyleSheet("color:#00d4ff;font-size:18px;font-weight:bold;")
        hb.addWidget(t)
       
        x = QPushButton("✖")
        x.setFixedSize(30,30)
        x.setStyleSheet("QPushButton{background:#e74c3c;color:white;border:none;border-radius:15px;font-size:16px;} QPushButton:hover{background:#c0392b;}")
        x.clicked.connect(self.toggle_chat)
        hb.addWidget(x)
        v.addWidget(hdr)
       
        self.chat_tabs = QTabWidget()
        self.chat_tabs.setStyleSheet("""
            QTabWidget::pane{border:1px solid #444;border-radius:5px;background:rgba(15,20,30,0.5);}
            QTabBar::tab{background:#2a3942;color:white;padding:8px 15px;margin:2px;border-radius:5px;}
            QTabBar::tab:selected{background:#0078d4;}
        """)
        self._ensure_tab("Everyone")
        v.addWidget(self.chat_tabs)
       
        box = QWidget()
        vb = QVBoxLayout(box)
        vb.setContentsMargins(0,10,0,0)
        vb.setSpacing(5)
       
        rb = QHBoxLayout()
        rlab = QLabel("To:")
        rlab.setStyleSheet("color:#aaa;font-size:12px;")
        rb.addWidget(rlab)
       
        self.recipient_combo = QComboBox()
        self.recipient_combo.addItem("Everyone")
        self.recipient_combo.setStyleSheet("QComboBox{background:#2a3942;color:white;border:1px solid #0078d4;border-radius:5px;padding:5px;}")
        self.recipient_combo.currentTextChanged.connect(self._switch_tab_to_recipient)
        rb.addWidget(self.recipient_combo)
        vb.addLayout(rb)
       
        ib = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type a message...")
        self.chat_input.setStyleSheet("""
            QLineEdit{background:#2a3942;color:white;border:2px solid #0078d4;border-radius:22px;padding:12px 18px;font-size:14px;}
            QLineEdit:focus{border:2px solid #00ff88;}
        """)
        self.chat_input.returnPressed.connect(self.send_chat)
       
        send = QPushButton("➤")
        send.setFixedSize(44,44)
        send.setStyleSheet("QPushButton{background:#00a884;color:white;border:none;border-radius:22px;font-size:20px;font-weight:bold;} QPushButton:hover{background:#00ff88;}")
        send.clicked.connect(self.send_chat)
       
        ib.addWidget(self.chat_input)
        ib.addWidget(send)
        vb.addLayout(ib)
        v.addWidget(box)
       
        return panel

    def _switch_tab_to_recipient(self, name: str):
        for i in range(self.chat_tabs.count()):
            if self.chat_tabs.tabText(i) == name:
                self.chat_tabs.setCurrentIndex(i)
                return

    def _ensure_tab(self, key: str):
        if key in self.chat_containers:
            return
       
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setStyleSheet("background:transparent;border:none;")
       
        cont = QWidget()
        vl = QVBoxLayout(cont)
        vl.addStretch()
        vl.setSpacing(12)
        vl.setContentsMargins(10,10,10,10)
        sc.setWidget(cont)
       
        self.chat_containers[key] = (cont, vl, sc)
        self.chat_tabs.addTab(sc, key)

    def _build_files_panel(self):
        panel = QFrame()
        panel.setStyleSheet("QFrame{background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1e2749, stop:1 #1a1a2e); border:2px solid #00d4ff; border-radius:15px;}")
        v = QVBoxLayout(panel)
        v.setContentsMargins(15,15,15,15)
       
        hdr = QWidget()
        hb = QHBoxLayout(hdr)
        hb.setContentsMargins(0,0,0,10)
       
        t = QLabel("📁 Shared Files")
        t.setStyleSheet("color:#00d4ff;font-size:18px;font-weight:bold;")
        hb.addWidget(t)
       
        x = QPushButton("✖")
        x.setFixedSize(30,30)
        x.setStyleSheet("QPushButton{background:#e74c3c;color:white;border:none;border-radius:15px;font-size:16px;} QPushButton:hover{background:#c0392b;}")
        x.clicked.connect(self.toggle_files)
        hb.addWidget(x)
        v.addWidget(hdr)
       
        up = QPushButton("⬆️ Upload File(s)")
        up.setStyleSheet("QPushButton{background:#0078d4;color:white;border:none;border-radius:8px;padding:10px;font-size:14px;} QPushButton:hover{background:#005a9e;}")
        up.clicked.connect(self.upload_files)
        v.addWidget(up)
       
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            QListWidget{background-color:rgba(15,20,30,0.5);border:1px solid #444;border-radius:10px;color:white;padding:5px;}
            QListWidget::item{padding:10px;border-radius:5px;margin:2px;}
            QListWidget::item:hover{background:#2a3942;}
            QListWidget::item:selected{background:#0078d4;}
        """)
        self.file_list.itemDoubleClicked.connect(self._download_clicked)
        v.addWidget(self.file_list)
       
        hint = QLabel("💡 Double-click to download")
        hint.setStyleSheet("color:#888;font-size:11px;padding:5px;")
        v.addWidget(hint)
       
        return panel

    def _build_participants_panel(self):
        panel = QFrame()
        panel.setStyleSheet("QFrame{background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1e2749, stop:1 #1a1a2e); border:2px solid #00d4ff; border-radius:15px;}")
        v = QVBoxLayout(panel)
        v.setContentsMargins(15,15,15,15)
       
        hdr = QWidget()
        hb = QHBoxLayout(hdr)
        hb.setContentsMargins(0,0,0,10)
       
        t = QLabel("👥 Participants")
        t.setStyleSheet("color:#00d4ff;font-size:18px;font-weight:bold;")
        hb.addWidget(t)
       
        x = QPushButton("✖")
        x.setFixedSize(30,30)
        x.setStyleSheet("QPushButton{background:#e74c3c;color:white;border:none;border-radius:15px;font-size:16px;} QPushButton:hover{background:#c0392b;}")
        x.clicked.connect(self.toggle_participants)
        hb.addWidget(x)
        v.addWidget(hdr)
       
        self.participant_list = QListWidget()
        self.participant_list.setStyleSheet("""
            QListWidget{background-color:rgba(15,20,30,0.5);border:1px solid #444;border-radius:10px;color:white;padding:5px;}
            QListWidget::item{padding:12px;border-radius:5px;margin:3px;font-size:14px;}
            QListWidget::item:hover{background:#2a3942;}
        """)
        v.addWidget(self.participant_list)
       
        return panel

    def _start_meeting_timer(self):
        self.meeting_timer = QTimer()
        self.meeting_timer.timeout.connect(self._update_meeting_timer)
        self.meeting_timer.start(1000)

    def _update_meeting_timer(self):
        try:
            elapsed = datetime.now() - self.meeting_start_time
            hours = int(elapsed.total_seconds() // 3600)
            minutes = int((elapsed.total_seconds() % 3600) // 60)
            seconds = int(elapsed.total_seconds() % 60)
            self.timer_label.setText(f"⏱️ {hours:02d}:{minutes:02d}:{seconds:02d}")
        except:
            pass

    def _start_stream_timeout_checker(self):
        self.stream_timeout_timer = QTimer()
        self.stream_timeout_timer.timeout.connect(self._check_stream_timeouts)
        self.stream_timeout_timer.start(2000)

    def _check_stream_timeouts(self):
        try:
            locker = QMutexLocker(self.video_mutex)
            now = datetime.now()
           
            # Don't timeout streams during screen sharing - they're just hidden
            if self.viewing_screen or self.screen_active:
                return
           
            timeout_duration = timedelta(seconds=10)
           
            stale_ssrcs = []
            for ssrc, last_time in list(self.stream_last_frame_time.items()):
                if now - last_time > timeout_duration:
                    stale_ssrcs.append(ssrc)
           
            for ssrc in stale_ssrcs:
                logger.info(f"Removing stale video stream: {ssrc}")
                self._remove_video_stream(ssrc)
        except Exception as e:
            logger.error(f"Stream timeout check error: {e}")

    def _remove_video_stream(self, ssrc: int):
        """Remove a video stream but keep participant slot."""
        try:
            ssrc_unsigned = ssrc & 0xFFFFFFFF
           
            # Find username for this SSRC
            username = self.ssrc_to_username.get(ssrc_unsigned)
           
            if username and username in self.participant_slots:
                # Clear video but keep slot
                label = self.participant_slots[username]
                label.clear()
                label.setPixmap(QPixmap())
                label.setText(f"👤 {username}\n(Camera Off)")
                label.update()
               
                # Remove SSRC mapping but keep slot
                if ssrc_unsigned in self.video_labels:
                    del self.video_labels[ssrc_unsigned]
                if username in self.username_to_ssrc:
                    del self.username_to_ssrc[username]
                if ssrc_unsigned in self.ssrc_to_username:
                    del self.ssrc_to_username[ssrc_unsigned]
               
                logger.info(f"Cleared video for {username}, slot retained")
           
            # Clean up timestamp tracking
            if ssrc_unsigned in self.stream_last_frame_time:
                del self.stream_last_frame_time[ssrc_unsigned]
               
        except Exception as e:
            logger.error(f"Error removing video stream: {e}")

    def _rebuild_video_grid(self):
        """Rebuild video grid with dynamic sizing."""
        try:
            if self.screen_active or self.viewing_screen:
                return
           
            locker = QMutexLocker(self.video_mutex)
           
            # Clear existing layout
            while self.video_layout.count():
                child = self.video_layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)
                elif child.layout():
                    self._clear_layout(child.layout())
           
            # Show all video labels (OTHER users only)
            for label in self.video_labels.values():
                label.show()
           
            # Show all participant slots instead of just active video streams
            active_participants = list(self.participant_slots.keys())
            num_videos = len(active_participants)
           
            if num_videos == 0:
                # Only self-view, ensure it's visible
                self.self_view.raise_()
                self.self_view.show()
                return
           
            if self.pinned_ssrc:
                self._build_pinned_layout()
            else:
                self._build_dynamic_grid(num_videos, active_participants)
           
            # ALWAYS ensure self view is on top in corner
            self.self_view.raise_()
            self.self_view.show()
            self.self_view.move(20, 20)
        except Exception as e:
            logger.error(f"Grid rebuild error: {e}")

    def _build_dynamic_grid(self, num_videos: int, ssrcs: list):
        """Build dynamic grid that fills the screen based on participant count."""
        try:
            # Calculate optimal grid layout
            if num_videos == 1:
                # Single video - full screen
                cols, rows = 1, 1
            elif num_videos == 2:
                # Two videos - side by side
                cols, rows = 2, 1
            elif num_videos <= 4:
                # Up to 4 videos - 2x2 grid
                cols, rows = 2, 2
            elif num_videos <= 6:
                # Up to 6 videos - 3x2 grid
                cols, rows = 3, 2
            elif num_videos <= 9:
                # Up to 9 videos - 3x3 grid
                cols, rows = 3, 3
            else:
                # More than 9 - calculate grid
                cols = math.ceil(math.sqrt(num_videos))
                rows = math.ceil(num_videos / cols)
           
            grid = QGridLayout()
            grid.setSpacing(15)
            grid.setContentsMargins(0, 0, 0, 0)
           
            for idx, username in enumerate(ssrcs):
                if username in self.participant_slots:
                    label = self.participant_slots[username]
                    row = idx // cols
                    col = idx % cols
                   
                    # Remove size constraints to allow dynamic sizing
                    label.setMinimumSize(100, 75)
                    label.setMaximumSize(16777215, 16777215)  # Qt max size
                   
                    grid.addWidget(label, row, col)
                   
                    # Set equal stretch for all rows and columns
                    grid.setRowStretch(row, 1)
                    grid.setColumnStretch(col, 1)
           
            self.video_layout.addLayout(grid)
        except Exception as e:
            logger.error(f"Dynamic grid error: {e}")

    def _clear_layout(self, layout):
        """Recursively clear a layout."""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
            elif child.layout():
                self._clear_layout(child.layout())

    def _build_grid_layout(self, num_videos: int, ssrcs: list):
        try:
            cols = math.ceil(math.sqrt(num_videos))
            rows = math.ceil(num_videos / cols)
           
            grid = QGridLayout()
            grid.setSpacing(15)
           
            for idx, ssrc in enumerate(ssrcs):
                if ssrc in self.video_labels:
                    row = idx // cols
                    col = idx % cols
                    grid.addWidget(self.video_labels[ssrc], row, col)
           
            self.video_layout.addLayout(grid)
        except:
            pass

    def _build_pinned_layout(self):
        try:
            hbox = QHBoxLayout()
            hbox.setSpacing(15)
           
            if self.pinned_ssrc in self.participant_slots:
                pinned_label = self.participant_slots[self.pinned_ssrc]
                hbox.addWidget(pinned_label, stretch=3)

            others = [username for username in self.participant_slots.keys() if username != self.pinned_ssrc]
           
            if others:
                vbox = QVBoxLayout()
                vbox.setSpacing(15)
               
                for username in others:
                    if username in self.participant_slots:
                        vbox.addWidget(self.participant_slots[username])
               
                vbox.addStretch()
                hbox.addLayout(vbox, stretch=1)
           
            self.video_layout.addLayout(hbox)
        except:
            pass

    def pin_video(self, identifier):
        """Pin video by username or SSRC."""
        # identifier can be username or "SELF"
        self.pinned_ssrc = identifier
        self._rebuild_video_grid()

    def unpin_video(self):
        self.pinned_ssrc = None
        self._rebuild_video_grid()

    def toggle_chat(self):
        self.chat_visible = not self.chat_visible
        if self.chat_visible:
            self.files_panel.hide()
            self.participants_panel.hide()
            self.chat_panel.show()
        else:
            self.chat_panel.hide()

    def toggle_files(self):
        self.files_visible = not self.files_visible
        if self.files_visible:
            self.chat_panel.hide()
            self.participants_panel.hide()
            self.files_panel.show()
            self._refresh_file_list()
        else:
            self.files_panel.hide()

    def toggle_participants(self):
        self.participants_visible = not self.participants_visible
        if self.participants_visible:
            self.chat_panel.hide()
            self.files_panel.hide()
            self.participants_panel.show()
            self._update_participant_list()
        else:
            self.participants_panel.hide()

    def _update_participant_list(self):
        try:
            self.participant_list.clear()
           
            my_status = f"👤 {self.my_username} (You)"
            if self.mic_active:
                my_status += " 🎤"
            if self.camera_active:
                my_status += " 📹"
           
            item = QListWidgetItem(my_status)
            self.participant_list.addItem(item)
           
            for user in self.online_users:
                username = user.get("username", "")
                if username and username != self.my_username:
                    item = QListWidgetItem(f"👤 {username}")
                    self.participant_list.addItem(item)
        except:
            pass

    def _refresh_file_list(self):
        def _refresh():
            try:
                files = self.file_client.list_files()
                for file_info in files:
                    fid = file_info.get("file_id")
                    fname = file_info.get("filename")
                    fsize = file_info.get("size", 0)
                    if fid and fid not in self.available_files:
                        self.file_available.emit(fid, fname, fsize)
            except:
                pass
       
        threading.Thread(target=_refresh, daemon=True).start()

    def upload_files(self):
        names, _ = QFileDialog.getOpenFileNames(self, "Select File(s) to Upload")
        if names:
            for name in names:
                threading.Thread(target=self._do_upload, args=(name,), daemon=True).start()

    def _do_upload(self, path: str):
        try:
            fid = self.file_client.upload_file(path)
            if fid:
                fname = Path(path).name
                self.chat_received.emit("System", f"📤 You uploaded: {fname}", False, "Everyone")
                self._refresh_file_list()
        except:
            pass

    def _download_clicked(self, item: QListWidgetItem):
        fid = item.data(Qt.ItemDataRole.UserRole)
        if not fid or fid not in self.available_files:
            return
       
        meta = self.available_files[fid]
        fname = meta["filename"]
        save_path, _ = QFileDialog.getSaveFileName(self, "Save As", fname)
       
        if save_path:
            threading.Thread(target=self._do_download, args=(fid, save_path, fname), daemon=True).start()

    def _do_download(self, fid: str, save_path: str, fname: str):
        try:
            success = self.file_client.download_file(fid, save_path)
            if success:
                self.chat_received.emit("System", f"📥 Downloaded: {fname}", False, "Everyone")
        except:
            pass

    def _setup_file_client(self):
        self.file_client = FileClient(self.config)

    def _setup_media(self):
        self.video_receiver = VideoReceiver(self.config, frame_callback=self._on_video_frame)
        self.video_receiver.start()
        self.video_sender = VideoSender(self.config, shared_socket=self.video_receiver.get_socket())
       
        self.audio_receiver = AudioReceiver(self.config)
        self.audio_receiver.start()
        self.audio_sender = AudioSender(self.config, shared_socket=self.audio_receiver.get_socket())
       
        self.screen_sharer = ScreenSharer(self.config)
        self.screen_viewer = ScreenViewer(self.config, frame_callback=self._on_screen_frame)
        self.screen_viewer.start()

    def _start_control(self):
        self.control = ControlClient(self.config, gui_callback=self._on_control_event)
        threading.Thread(target=self._connect, args=(self.my_username,), daemon=True).start()

    def _connect(self, uname: str):
        self.control.connect(uname)

    def _on_control_event(self, etype: str, payload: dict):
        if etype == "chat":
            f = payload.get("from","")
            txt = payload.get("text","")
            to = payload.get("to", "Everyone")
            is_self = (f == self.my_username)
            if not is_self:
                tab = "Everyone" if to == "Everyone" else f
                self.chat_received.emit(f, txt, is_self, tab)
        elif etype == "roster":
            self.roster_updated.emit(payload.get("users", []))
        elif etype == "file_available":
            self.file_available.emit(payload.get("file_id",""), payload.get("filename",""), payload.get("size",0))

    def send_chat(self):
        txt = self.chat_input.text().strip()
        if not txt or not self.control or not self.control.connected:
            return
       
        rcpt = self.recipient_combo.currentText()
        target = rcpt if rcpt != "Everyone" else None
        self.control.send_chat(txt, target)
       
        tab = rcpt
        self.chat_received.emit(self.my_username, txt, True, tab)
        self.chat_input.clear()

    @pyqtSlot(str, str, bool, str)
    def _add_chat_message_slot(self, username: str, text: str, is_self: bool, tab_key: str):
        try:
            self._ensure_tab(tab_key)
            cont, v, sc = self.chat_containers[tab_key]
           
            w = QWidget()
            vb = QVBoxLayout(w)
            vb.setContentsMargins(5,2,5,2)
           
            user_label = QLabel(username)
            user_label.setStyleSheet(f"color:{'#00ff88' if is_self else '#0078d4'};font-size:11px;font-weight:bold;")
           
            bubble = QLabel(text)
            bubble.setWordWrap(True)
            bubble.setMaximumWidth(260)
            bubble.setStyleSheet(f"QLabel{{background:{'#005c4b' if is_self else '#1f2c33'};color:white;border-radius:10px;padding:10px 14px;font-size:13px;}}")
           
            if is_self:
                user_label.setAlignment(Qt.AlignmentFlag.AlignRight)
                vb.addWidget(user_label)
                vb.addWidget(bubble, alignment=Qt.AlignmentFlag.AlignRight)
            else:
                vb.addWidget(user_label)
                vb.addWidget(bubble)
           
            v.takeAt(v.count()-1)
            v.addWidget(w)
            v.addStretch()
           
            QTimer.singleShot(50, lambda: sc.verticalScrollBar().setValue(sc.verticalScrollBar().maximum()))
        except:
            pass

    @pyqtSlot(list)
    def _on_roster_slot(self, users):
        try:
            self.online_users = users
            locker = QMutexLocker(self.video_mutex)
           
            # Get current usernames from roster (excluding self)
            current_usernames = {u.get("username") for u in users
                            if u.get("username") and u.get("username") != self.my_username}
           
            # Remove slots for users who left
            for username in list(self.participant_slots.keys()):
                if username not in current_usernames:
                    self._remove_participant_slot(username)
           
            # Add slots for new users
            for user in users:
                username = user.get("username")
                if not username or username == self.my_username:
                    continue
               
                if username not in self.participant_slots:
                    self._create_participant_slot(username)
           
            # Update chat recipient combo
            names = list(current_usernames)
            cur = self.recipient_combo.currentText()
            self.recipient_combo.blockSignals(True)
            self.recipient_combo.clear()
            self.recipient_combo.addItem("Everyone")
           
            for n in names:
                self.recipient_combo.addItem(n)
                self._ensure_tab(n)
           
            self.recipient_combo.blockSignals(False)
            idx = self.recipient_combo.findText(cur)
            if idx >= 0:
                self.recipient_combo.setCurrentIndex(idx)
           
            # Rebuild video grid if not sharing screen
            if not self.screen_active and not self.viewing_screen:
                QTimer.singleShot(100, self._rebuild_video_grid)
           
            if self.participants_visible:
                self._update_participant_list()
        except Exception as e:
            logger.error(f"Roster update error: {e}")

    def _create_participant_slot(self, username: str):
        """Create a persistent slot for a participant."""
        try:
            label = VideoLabel(ssrc=None, parent_window=self)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("""
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2d3561, stop:1 #1e2749);
                border: 2px solid #0078d4; border-radius: 15px;
                color: white; font-size: 16px; font-weight: bold;
            """)
            label.setMinimumSize(100, 75)
            label.setScaledContents(True)
            label.setText(f"👤 {username}\n(Camera Off)")
           
            self.participant_slots[username] = label
           
            # If screen sharing, create thumbnail too
            if self.screen_active:
                self.screen_thumbnails[username] = label
           
            logger.info(f"Created slot for participant: {username}")
        except Exception as e:
            logger.error(f"Error creating participant slot: {e}")

    def _remove_participant_slot(self, username: str):
        """Remove a participant's slot."""
        try:
            if username in self.participant_slots:
                label = self.participant_slots[username]
               
                # Clear the label
                label.clear()
                label.setPixmap(QPixmap())
                label.update()
               
                # Remove from layout
                label.setParent(None)
                label.deleteLater()
               
                del self.participant_slots[username]
               
                # Clean up related mappings
                if username in self.username_to_ssrc:
                    ssrc = self.username_to_ssrc[username]
                    if ssrc in self.ssrc_to_username:
                        del self.ssrc_to_username[ssrc]
                    del self.username_to_ssrc[username]
                   
                    # Clean up video labels
                    if ssrc in self.video_labels:
                        del self.video_labels[ssrc]
                   
                    if ssrc in self.stream_last_frame_time:
                        del self.stream_last_frame_time[ssrc]
               
                # Clean up screen thumbnails
                if username in self.screen_thumbnails:
                    del self.screen_thumbnails[username]
               
                if self.pinned_ssrc and self.pinned_ssrc == username:
                    self.pinned_ssrc = None
               
                logger.info(f"Removed slot for participant: {username}")
        except Exception as e:
            logger.error(f"Error removing participant slot: {e}")

    @pyqtSlot(str, str, int)
    def _on_file_available_slot(self, fid: str, fname: str, size: int):
        try:
            if not fid or fid in self.available_files:
                return
           
            self.available_files[fid] = {"filename": fname, "size": size}
            item = QListWidgetItem(f"📄 {fname} ({size} bytes)")
            item.setData(Qt.ItemDataRole.UserRole, fid)
            self.file_list.addItem(item)
        except:
            pass

    def _on_video_frame(self, ssrc: int, frame: np.ndarray):
        self.frame_received.emit(ssrc, frame)

    def _on_screen_frame(self, img, width, height):
        self.screen_received.emit(img, width, height)

    def _normalize_ssrc(self, ssrc: int) -> int:
        """Normalize SSRC to unsigned 32-bit integer."""
        return ssrc & 0xFFFFFFFF

    @pyqtSlot(int, np.ndarray)
    def _update_video_display(self, ssrc: int, frame: np.ndarray):
        """Update video display."""
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                return
               
            locker = QMutexLocker(self.video_mutex)
           
            ssrc_unsigned = ssrc & 0xFFFFFFFF
            self.stream_last_frame_time[ssrc_unsigned] = datetime.now()
           
            # Handle self video - ONLY update self view, don't create regular label
            if self.my_ssrc and ssrc_unsigned == (self.my_ssrc & 0xFFFFFFFF):
                # Update self view with received frame
                if self.self_view and self.self_view.isVisible():
                    self._blit(frame, self.self_view)
                # RETURN HERE - don't create a regular video label for own stream
                return
           
            # Map SSRC to existing participant slot
            if ssrc_unsigned not in self.video_labels:
                # Try to find matching participant slot
                username = self.ssrc_to_username.get(ssrc_unsigned)
               
                if username and username in self.participant_slots:
                    # Use existing slot
                    label = self.participant_slots[username]
                    self.video_labels[ssrc_unsigned] = label
                    self.username_to_ssrc[username] = ssrc_unsigned
                    logger.info(f"Mapped SSRC {ssrc_unsigned} to existing slot: {username}")
                else:
                    # SSRC without known username - try to match with roster
                    matched = False
                    for participant_username in self.participant_slots.keys():
                        if participant_username not in self.username_to_ssrc:
                            # This participant doesn't have an SSRC yet
                            label = self.participant_slots[participant_username]
                            self.video_labels[ssrc_unsigned] = label
                            self.username_to_ssrc[participant_username] = ssrc_unsigned
                            self.ssrc_to_username[ssrc_unsigned] = participant_username
                            matched = True
                            logger.info(f"Auto-mapped SSRC {ssrc_unsigned} to slot: {participant_username}")
                            break
                   
                    if not matched:
                        # Fallback: create temporary label (shouldn't happen with roster)
                        logger.warning(f"Received video from unknown SSRC: {ssrc_unsigned}")
                        return
               
                # Handle visibility based on screen sharing state
                if not self.screen_active:
                    if self.viewing_screen and not self.screen_presenting_ssrc:
                        self.screen_presenting_ssrc = ssrc_unsigned
                   
                    if self.viewing_screen:
                        label.hide()
                    else:
                        QTimer.singleShot(50, self._rebuild_video_grid)
           
            # Always update frame for other users - even if hidden
            if ssrc_unsigned in self.video_labels:
                label = self.video_labels[ssrc_unsigned]
                self._blit(frame, label)
               
                # Update thumbnail if screen sharing
                if self.screen_active and ssrc_unsigned in self.screen_thumbnails:
                    thumb = self.screen_thumbnails[ssrc_unsigned]
                    if thumb and thumb.isVisible():
                        self._blit(frame, thumb)
               
                # Update presenter video if viewing screen
                if self.viewing_screen and ssrc_unsigned == self.screen_presenting_ssrc:
                    self.presenter_last_frame = frame.copy()
                    if self.screen_presenter_video and self.screen_presenter_video.isVisible():
                        self._blit(frame, self.screen_presenter_video)
        except Exception as e:
            logger.error(f"Video display error: {e}")

    def _handle_critical_error(self, error_msg: str):
        """Handle critical errors that might crash the client."""
        logger.error(f"CRITICAL ERROR: {error_msg}")
       
        try:
            # Try to stop all media gracefully
            if self.camera_active:
                self.video_sender.stop()
                self.camera_active = False
           
            if self.mic_active:
                self.audio_sender.stop()
                self.mic_active = False
           
            if self.screen_active:
                self.screen_sharer.stop()
                self.screen_active = False
           
            # Show error to user
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error",
                            f"A critical error occurred:\n{error_msg}\n\nPlease restart the application.")
        except:
            pass

    @pyqtSlot(object, int, int)
    def _update_screen_display(self, img, width, height):
        """Update screen display."""
        try:
            locker = QMutexLocker(self.screen_mutex)
           
            if img is None:
                # Clean up screen sharing UI
                if self.screen_label:
                    self.screen_label.hide()
                    self.screen_label.deleteLater()
                    self.screen_label = None
               
                if self.screen_thumbnail_container:
                    for thumb in self.screen_thumbnails.values():
                        thumb.deleteLater()
                    self.screen_thumbnails.clear()
                    self.screen_thumbnail_container.deleteLater()
                    self.screen_thumbnail_container = None
               
                if self.screen_presenter_video:
                    self.screen_presenter_video.hide()
                    self.screen_presenter_video.deleteLater()
                    self.screen_presenter_video = None
               
                self.screen_presenting_ssrc = None
                self.presenter_last_frame = None
                self.viewing_screen = False
               
                # Show all video labels again
                for label in self.video_labels.values():
                    label.show()
               
                # Show self view and rebuild grid
                self.self_view.show()
                self.self_view.raise_()
               
                QTimer.singleShot(200, self._rebuild_video_grid)
                return
           
            frame = np.array(img)
            self.viewing_screen = True
           
            # Hide all video labels when viewing screen
            for label in self.video_labels.values():
                label.hide()
           
            # Hide self view when viewing screen
            self.self_view.hide()
           
            if not self.screen_label:
                self.screen_label = QLabel(self.video_container)
                self.screen_label.setStyleSheet("""
                    background: black;
                    border: 4px solid #00ff88;
                    border-radius: 15px;
                """)
                self.screen_label.setScaledContents(True)
           
            # Position screen label to fill container
            container_width = self.video_container.width() - 30
            container_height = self.video_container.height() - 30
           
            self.screen_label.setGeometry(15, 15, container_width, container_height)
            self.screen_label.show()
            self.screen_label.raise_()
            self._blit(frame, self.screen_label)
           
            if self.screen_active:
                # Show thumbnails when YOU are sharing
                QTimer.singleShot(100, self._update_screen_thumbnails)
                if self.screen_presenter_video:
                    self.screen_presenter_video.hide()
            else:
                # Show presenter video when VIEWING someone else's share
                self._ensure_presenter_video()
           
            logger.debug("Screen frame displayed")
        except Exception as e:
            logger.error(f"Screen display error: {e}")
            import traceback
            traceback.print_exc()

    def _create_video_label_with_placeholder(self, ssrc: int, username: str = "Participant"):
        """Create video label with placeholder."""
        label = VideoLabel(ssrc=ssrc, parent_window=self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2d3561, stop:1 #1e2749);
            border: 2px solid #0078d4; border-radius: 15px;
            color: white; font-size: 18px; font-weight: bold;
        """)
        label.setMinimumSize(100, 75)
        label.setScaledContents(True)
        label.setText(f"👤 {username}\n(Connecting...)")

    def _ensure_presenter_video(self):
        """Ensure presenter video widget exists."""
        try:
            if not self.screen_presenter_video:
                self.screen_presenter_video = QLabel(self.video_container)
                self.screen_presenter_video.setFixedSize(220, 165)
                self.screen_presenter_video.setStyleSheet("""
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #00d4ff, stop:1 #0078d4);
                    border: 3px solid #ff6b6b; border-radius: 12px;
                """)
                self.screen_presenter_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.screen_presenter_video.setScaledContents(True)
                self.screen_presenter_video.setText("📺 Presenter")
           
            if self.presenter_last_frame is not None:
                self._blit(self.presenter_last_frame, self.screen_presenter_video)
            elif self.screen_presenting_ssrc and self.screen_presenting_ssrc in self.video_labels:
                presenter_label = self.video_labels[self.screen_presenting_ssrc]
                if presenter_label.pixmap():
                    self.screen_presenter_video.setPixmap(presenter_label.pixmap())
           
            self.screen_presenter_video.show()
            self.screen_presenter_video.raise_()
           
            cw, ch = self.video_container.width(), self.video_container.height()
            self.screen_presenter_video.move(cw - self.screen_presenter_video.width() - 35,
                                             ch - self.screen_presenter_video.height() - 35)
        except:
            pass

    def _update_screen_thumbnails(self):
        """Update thumbnail container."""
        try:
            if not self.screen_active:
                return
           
            locker = QMutexLocker(self.video_mutex)
           
            if not self.screen_thumbnail_container:
                self.screen_thumbnail_container = QFrame(self.video_container)
                self.screen_thumbnail_container.setStyleSheet("""
                    QFrame {
                        background: rgba(0,0,0,0.85);
                        border: 2px solid #0078d4;
                        border-radius: 10px;
                    }
                """)
                self.screen_thumbnail_layout = QVBoxLayout(self.screen_thumbnail_container)
                self.screen_thumbnail_layout.setSpacing(10)
                self.screen_thumbnail_layout.setContentsMargins(10,10,10,10)
           
            # Remove thumbnails for disconnected streams
            for ssrc in list(self.screen_thumbnails.keys()):
                if ssrc not in self.video_labels:
                    thumb = self.screen_thumbnails[ssrc]
                    self.screen_thumbnail_layout.removeWidget(thumb)
                    thumb.deleteLater()
                    del self.screen_thumbnails[ssrc]
           
            # Add thumbnails for new streams
            for ssrc, label in self.video_labels.items():
                if ssrc not in self.screen_thumbnails:
                    thumb = QLabel()
                    thumb.setFixedSize(220, 165)
                    thumb.setStyleSheet("""
                        background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2d3561, stop:1 #1e2749);
                        border: 2px solid #0078d4;
                        border-radius: 10px;
                    """)
                    thumb.setScaledContents(True)
                    thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    thumb.setText(f"👤 Participant")
                   
                    self.screen_thumbnail_layout.addWidget(thumb)
                    self.screen_thumbnails[ssrc] = thumb
           
            # Update thumbnail content
            for ssrc, thumb in self.screen_thumbnails.items():
                if ssrc in self.video_labels:
                    label = self.video_labels[ssrc]
                    if label.pixmap() and not label.pixmap().isNull():
                        thumb.setPixmap(label.pixmap())
                        thumb.setText("")
           
            # Add stretch at end if needed
            if self.screen_thumbnail_layout.count() > 0:
                last_item = self.screen_thumbnail_layout.itemAt(self.screen_thumbnail_layout.count() - 1)
                if not last_item or not last_item.spacerItem():
                    self.screen_thumbnail_layout.addStretch()
           
            # Position container
            cw, ch = self.video_container.width(), self.video_container.height()
            container_width = 250
            self.screen_thumbnail_container.setGeometry(cw - container_width - 15, 15,
                                                        container_width, ch - 30)
           
            self.screen_thumbnail_container.show()
            self.screen_thumbnail_container.raise_()
        except Exception as e:
            logger.error(f"Thumbnail update error: {e}")

    def _blit(self, frame: np.ndarray, lab: QLabel):
        """Blit frame to label."""
        try:
            if lab is None:
                return
           
            # Don't update if widget is not visible
            if not lab.isVisible():
                return
           
            # Create a copy to prevent memory issues
            frame = frame.copy()
           
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
            h, w = rgb.shape[:2]
            ch = rgb.shape[2] if len(rgb.shape) == 3 else 1
            bpl = ch * w
            rgb = np.ascontiguousarray(rgb)
           
            if ch == 3:
                img = QImage(rgb.data, w, h, bpl, QImage.Format.Format_RGB888).copy()
            else:
                img = QImage(rgb.data, w, h, bpl, QImage.Format.Format_Grayscale8).copy()
           
            if img.isNull():
                return
       
            pixmap = QPixmap.fromImage(img)
           
            # Update in main thread
            if threading.current_thread() is threading.main_thread():
                lab.setPixmap(pixmap)
                lab.setText("")
                lab.update()
            else:
                from PyQt6.QtCore import QMetaObject, Qt
                QMetaObject.invokeMethod(lab, "setPixmap", Qt.ConnectionType.QueuedConnection, pixmap)
                QMetaObject.invokeMethod(lab, "setText", Qt.ConnectionType.QueuedConnection, "")
               
        except Exception as e:
            logger.debug(f"Blit error: {e}")

    def toggle_camera(self):
        if not self.camera_active:
            threading.Thread(target=self._start_cam, daemon=True).start()
        else:
            self._stop_cam()

    def _start_cam(self):
        if self.video_sender.start():
            self.camera_active = True
            self.my_ssrc = self.video_sender.ssrc
           
            # Clear any old content
            self.self_view.clear()
            self.self_view.setPixmap(QPixmap())
            self.self_view.setText("")
           
            # Force visibility
            self.self_view.show()
            self.self_view.raise_()
            self.self_view.update()
           
            self.cam_btn.button.setStyleSheet("""
                QPushButton{background-color:#27ae60;color:white;border:none;border-radius:30px;font-size:24px;}
                QPushButton:hover{background-color:#27ae60;border:3px solid #00ff88;}
            """)
           
            # Don't start separate preview loop - rely on network feedback
            # This prevents double rendering and ensures you see what others see
            logger.info("Camera started")

    # def _self_preview_loop(self):
    #     frame_time = 1.0 / 30.0
       
    #     while self.preview_running and self.camera_active:
    #         try:
    #             if (self.video_sender and
    #                 hasattr(self.video_sender, 'last_frame') and
    #                 self.video_sender.last_frame is not None):
                   
    #                 frame = self.video_sender.last_frame
                   
    #                 # Only update if widget is visible
    #                 if self.self_view and self.self_view.isVisible():
    #                     self._blit(frame, self.self_view)
               
    #             time.sleep(frame_time)
    #         except Exception as e:
    #             logger.debug(f"Preview loop error: {e}")
    #             time.sleep(0.1)
    #             continue
   
    # logger.info("Self preview loop ended")

    def _stop_cam(self):
        """Stop camera."""
        self.camera_active = False
       
        # Stop video sender
        if self.video_sender:
            self.video_sender.stop()
            self.video_sender.last_frame = None
       
        # DON'T try to remove video label - we never created one for self
        self.my_ssrc = None
       
        # Clear self view in main thread
        QTimer.singleShot(0, self._clear_self_view)
       
        logger.info("Camera stopped")

    def _clear_self_view(self):
        """Clear self view in main thread."""
        try:
            if self.self_view:
                # Clear pixmap first
                self.self_view.clear()
                self.self_view.setPixmap(QPixmap())
               
                # Set text
                self.self_view.setText("📹 You\n(Camera Off)")
               
                # Force immediate update
                self.self_view.update()
                self.self_view.repaint()
               
                # Reset button style
                self.cam_btn.button.setStyleSheet(f"""
                    QPushButton{{background-color:{self.cam_btn.default_color};color:white;border:none;border-radius:30px;font-size:24px;}}
                    QPushButton:hover{{background-color:{self.cam_btn.default_color};border:3px solid #00ff88;}}
                """)
        except Exception as e:
            logger.error(f"Error clearing self view: {e}")

    def toggle_mic(self):
        if not self.mic_active:
            self._start_mic()
        else:
            self._stop_mic()

    def _start_mic(self):
        """Start microphone."""
        try:
            if self.audio_sender.start():
                self.mic_active = True
                logger.info("Mic started")
               
                # Update button color
                self.mic_btn.button.setStyleSheet("""
                    QPushButton{background-color:#27ae60;color:white;border:none;border-radius:30px;font-size:24px;}
                    QPushButton:hover{background-color:#27ae60;border:3px solid #00ff88;}
                """)
        except Exception as e:
            logger.error(f"Failed to start mic: {e}")

    def _stop_mic(self):
        """Stop microphone."""
        try:
            self.audio_sender.stop()
            self.mic_active = False
            logger.info("Mic stopped")
           
            # Reset button color
            self.mic_btn.button.setStyleSheet(f"""
                QPushButton{{background-color:{self.mic_btn.default_color};color:white;border:none;border-radius:30px;font-size:24px;}}
                QPushButton:hover{{background-color:{self.mic_btn.default_color};border:3px solid #00ff88;}}
            """)
        except Exception as e:
            logger.error(f"Failed to stop mic: {e}")

    def toggle_screen(self):
        """Toggle screen sharing."""
        if not self.screen_active:
            self._start_screen()
        else:
            self._stop_screen()

    def _start_screen(self):
        """Start screen sharing."""
        if self.screen_active:
            return  # Already active, prevent duplicate
       
        def start_thread():
            try:
                if self.screen_sharer.start(monitor=1, quality=50, fps=8):
                    self.screen_active = True
                    self.viewing_screen = False
                   
                    # Hide self view during screen sharing
                    self.self_view.hide()
                   
                    # Hide all video labels
                    locker = QMutexLocker(self.video_mutex)
                    for label in self.video_labels.values():
                        label.hide()
                   
                    # Update button style
                    self.scr_btn.button.setStyleSheet("""
                        QPushButton{background-color:#27ae60;color:white;border:none;border-radius:30px;font-size:24px;}
                        QPushButton:hover{background-color:#27ae60;border:3px solid #00ff88;}
                    """)
                   
                    logger.info("Screen sharing started")
                else:
                    logger.error("Failed to start screen sharing")
            except Exception as e:
                logger.error(f"Screen sharing error: {e}")
                import traceback
                traceback.print_exc()
       
        threading.Thread(target=start_thread, daemon=True).start()

    def _stop_screen(self):
        """Stop screen sharing."""
        try:
            self.screen_sharer.stop()
            self.screen_active = False
           
            # Clear screen display
            if self.screen_label:
                self.screen_label.hide()
                self.screen_label.deleteLater()
                self.screen_label = None
           
            # Clear thumbnails
            if self.screen_thumbnail_container:
                for thumb in self.screen_thumbnails.values():
                    thumb.deleteLater()
                self.screen_thumbnails.clear()
                self.screen_thumbnail_container.deleteLater()
                self.screen_thumbnail_container = None
           
            # Show self view
            self.self_view.show()
            self.self_view.raise_()
           
            # Show all video labels and rebuild grid
            for label in self.video_labels.values():
                label.show()
           
            # Reset button style
            self.scr_btn.button.setStyleSheet(f"""
                QPushButton{{background-color:{self.scr_btn.default_color};color:white;border:none;border-radius:30px;font-size:24px;}}
                QPushButton:hover{{background-color:{self.scr_btn.default_color};border:3px solid #00ff88;}}
            """)
           
            # Rebuild video grid
            QTimer.singleShot(100, self._rebuild_video_grid)
           
            logger.info("Screen sharing stopped")
        except Exception as e:
            logger.error(f"Error stopping screen share: {e}")

    def resizeEvent(self, e):
        """Handle window resize."""
        super().resizeEvent(e)
        try:
            # Position side panels
            panel_width = 370
            panel_x = self.width() - panel_width - 10
            panel_y = 80
            panel_height = self.height() - 170
           
            if hasattr(self, 'chat_panel') and self.chat_panel:
                self.chat_panel.setGeometry(panel_x, panel_y, panel_width, panel_height)
           
            if hasattr(self, 'files_panel') and self.files_panel:
                self.files_panel.setGeometry(panel_x, panel_y, panel_width, panel_height)
           
            if hasattr(self, 'participants_panel') and self.participants_panel:
                self.participants_panel.setGeometry(panel_x, panel_y, panel_width, panel_height)
           
            # Reposition self-view
            if hasattr(self, 'self_view') and self.self_view:
                self.self_view.move(20, 20)
                self.self_view.raise_()
           
            # Update screen label if showing
            if hasattr(self, 'screen_label') and self.screen_label and self.viewing_screen:
                cw, ch = self.video_container.width(), self.video_container.height()
                self.screen_label.setGeometry(15, 15, cw - 30, ch - 30)
           
            # Update screen thumbnails if showing
            if hasattr(self, 'screen_thumbnail_container') and self.screen_thumbnail_container and self.screen_active:
                cw, ch = self.video_container.width(), self.video_container.height()
                self.screen_thumbnail_container.setGeometry(cw - 265, 15, 250, ch - 30)
           
            # Update presenter video if showing
            if hasattr(self, 'screen_presenter_video') and self.screen_presenter_video and self.viewing_screen:
                cw, ch = self.video_container.width(), self.video_container.height()
                self.screen_presenter_video.move(cw - self.screen_presenter_video.width() - 35,
                                                ch - self.screen_presenter_video.height() - 35)
        except Exception as ex:
            logger.error(f"Resize error: {ex}")

    def closeEvent(self, e):
        """Clean shutdown."""
        try:
            logger.info("Shutting down client...")
           
            # Stop preview
            self.preview_running = False
           
            # Stop timers
            if hasattr(self, 'meeting_timer'):
                self.meeting_timer.stop()
            if hasattr(self, 'stream_timeout_timer'):
                self.stream_timeout_timer.stop()
           
            # Stop media
            if self.camera_active and self.video_sender:
                self.video_sender.stop()
                self.camera_active = False
           
            if self.mic_active and self.audio_sender:
                self.audio_sender.stop()
                self.mic_active = False
           
            if self.screen_active and self.screen_sharer:
                self.screen_sharer.stop()
                self.screen_active = False
           
            # Stop receivers
            if self.video_receiver:
                self.video_receiver.stop()
           
            if self.audio_receiver:
                self.audio_receiver.stop()
           
            if self.screen_viewer:
                self.screen_viewer.stop()
           
            # Disconnect control
            if self.control and self.control.connected:
                self.control.disconnect()
                # Give time for disconnect message
                import time
                time.sleep(0.1)
           
            logger.info("Client shutdown complete")
            e.accept()
        except Exception as ex:
            logger.error(f"Shutdown error: {ex}")
            e.accept()

    # ============================================================================
    # NEW FEATURE METHODS
    # ============================================================================
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for common actions."""
        # Mute/unmute mic
        QShortcut(QKeySequence("M"), self, self.toggle_mic)
        # Toggle camera
        QShortcut(QKeySequence("V"), self, self.toggle_camera)
        # Toggle screen share
        QShortcut(QKeySequence("S"), self, self.toggle_screen)
        # Raise hand
        QShortcut(QKeySequence("H"), self, self.toggle_raise_hand)
        # Toggle chat
        QShortcut(QKeySequence("C"), self, self.toggle_chat)
        # Fullscreen
        QShortcut(QKeySequence("F11"), self, self.toggle_fullscreen)
        logger.info("Keyboard shortcuts initialized: M=Mic, V=Camera, S=Screen, H=Hand, C=Chat, F11=Fullscreen")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.is_fullscreen:
            self.showNormal()
            self.is_fullscreen = False
        else:
            self.showFullScreen()
            self.is_fullscreen = True
    
    def toggle_raise_hand(self):
        """Toggle raise hand status."""
        try:
            self.hand_raised_state = not self.hand_raised_state
            
            # Update button appearance
            if self.hand_raised_state:
                self.hand_btn.button.setStyleSheet(f"""
                    QPushButton{{background-color:#ff9800;color:white;border:3px solid #ffeb3b;border-radius:30px;font-size:24px;}}
                    QPushButton:hover{{background-color:#ff9800;border:3px solid #00ff88;}}
                """)
                self.chat_received.emit("System", "✋ You raised your hand", False, "Everyone")
            else:
                self.hand_btn.button.setStyleSheet(f"""
                    QPushButton{{background-color:{self.hand_btn.default_color};color:white;border:none;border-radius:30px;font-size:24px;}}
                    QPushButton:hover{{background-color:{self.hand_btn.default_color};border:3px solid #00ff88;}}
                """)
                self.chat_received.emit("System", "✋ You lowered your hand", False, "Everyone")
            
            # Broadcast to server
            if self.control and self.control.connected:
                self.control.send_message({
                    "type": "hand_raise",
                    "username": self.my_username,
                    "raised": self.hand_raised_state
                })
        except Exception as e:
            logger.error(f"Error toggling raise hand: {e}")
    
    def _show_reaction_menu(self):
        """Show popup menu with emoji reactions."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 10px;
                padding: 10px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 8px 16px;
                font-size: 24px;
            }
            QMenu::item:selected {
                background-color: #444;
                border-radius: 5px;
            }
        """)
        
        for emoji in ReactionManager.REACTIONS:
            action = menu.addAction(emoji)
            action.triggered.connect(lambda checked, e=emoji: self._send_reaction(e))
        
        # Show menu at button position
        btn_pos = self.react_btn.button.mapToGlobal(QPoint(0, -200))
        menu.exec(btn_pos)
    
    def _send_reaction(self, emoji: str):
        """Send reaction emoji."""
        try:
            # Add to local reaction manager
            self.reaction_manager.add_reaction(emoji, self.my_username)
            
            # Show in chat
            self.chat_received.emit(self.my_username, emoji, False, "Everyone")
            
            # Broadcast to server
            if self.control and self.control.connected:
                self.control.send_message({
                    "type": "reaction",
                    "username": self.my_username,
                    "emoji": emoji
                })
            
            logger.info(f"Sent reaction: {emoji}")
        except Exception as e:
            logger.error(f"Error sending reaction: {e}")
    
    @pyqtSlot(str, str)
    def _on_reaction_received(self, emoji: str, username: str):
        """Handle reaction from another user."""
        try:
            self.reaction_manager.add_reaction(emoji, username)
            if username != self.my_username:
                self.chat_received.emit(username, emoji, False, "Everyone")
        except Exception as e:
            logger.error(f"Error handling reaction: {e}")
    
    @pyqtSlot(str, bool)
    def _on_hand_raised(self, username: str, raised: bool):
        """Handle hand raise from another user."""
        try:
            self.raised_hands[username] = raised
            status = "raised their hand ✋" if raised else "lowered their hand"
            self.chat_received.emit("System", f"{username} {status}", False, "Everyone")
        except Exception as e:
            logger.error(f"Error handling hand raise: {e}")
    
    def toggle_recording(self):
        """Toggle meeting recording."""
        try:
            if not self.is_recording:
                # Start recording
                output_path = self.meeting_recorder.start(f"meeting_{self.my_username}")
                if output_path:
                    self.is_recording = True
                    self.record_btn.button.setStyleSheet(f"""
                        QPushButton{{background-color:#ff0000;color:white;border:3px solid #ff6666;border-radius:30px;font-size:24px;}}
                        QPushButton:hover{{background-color:#ff0000;border:3px solid #00ff88;}}
                    """)
                    self.chat_received.emit("System", f"🔴 Recording started: {output_path}", False, "Everyone")
                    logger.info(f"Recording started: {output_path}")
            else:
                # Stop recording
                output_path = self.meeting_recorder.stop()
                if output_path:
                    self.is_recording = False
                    self.record_btn.button.setStyleSheet(f"""
                        QPushButton{{background-color:{self.record_btn.default_color};color:white;border:none;border-radius:30px;font-size:24px;}}
                        QPushButton:hover{{background-color:{self.record_btn.default_color};border:3px solid #00ff88;}}
                    """)
                    self.chat_received.emit("System", f"⏹️ Recording saved: {output_path}", False, "Everyone")
                    logger.info(f"Recording stopped: {output_path}")
        except Exception as e:
            logger.error(f"Error toggling recording: {e}")
    
    def _show_settings(self):
        """Show settings dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        dialog.setFixedSize(400, 500)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
                color: white;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #00ff88;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel { color: #ddd; }
            QComboBox, QSlider, QCheckBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        # Video Quality
        video_group = QGroupBox("Video Quality")
        video_layout = QVBoxLayout(video_group)
        quality_combo = QComboBox()
        quality_combo.addItems(["Low (320x240)", "Medium (640x480)", "High (1280x720)"])
        quality_combo.setCurrentIndex(1)  # Default medium
        video_layout.addWidget(QLabel("Resolution:"))
        video_layout.addWidget(quality_combo)
        layout.addWidget(video_group)
        
        # Virtual Background
        bg_group = QGroupBox("Virtual Background")
        bg_layout = QVBoxLayout(bg_group)
        
        bg_none = QCheckBox("None")
        bg_none.setChecked(True)
        bg_blur = QCheckBox("Blur Background")
        bg_custom = QCheckBox("Custom Image")
        
        bg_layout.addWidget(bg_none)
        bg_layout.addWidget(bg_blur)
        bg_layout.addWidget(bg_custom)
        
        def on_bg_change():
            if bg_blur.isChecked():
                self.virtual_bg.set_mode("blur")
            elif bg_custom.isChecked():
                # Open file dialog
                path, _ = QFileDialog.getOpenFileName(dialog, "Select Background", "", "Images (*.png *.jpg)")
                if path:
                    self.virtual_bg.set_mode("image", path)
            else:
                self.virtual_bg.set_mode("none")
        
        bg_blur.clicked.connect(on_bg_change)
        bg_custom.clicked.connect(on_bg_change)
        bg_none.clicked.connect(on_bg_change)
        layout.addWidget(bg_group)
        
        # Audio
        audio_group = QGroupBox("Audio")
        audio_layout = QVBoxLayout(audio_group)
        
        noise_check = QCheckBox("Noise Suppression")
        noise_check.setChecked(self.noise_suppressor.enabled)
        noise_check.stateChanged.connect(lambda s: setattr(self.noise_suppressor, 'enabled', s == Qt.CheckState.Checked.value))
        audio_layout.addWidget(noise_check)
        layout.addWidget(audio_group)
        
        # Keyboard Shortcuts Info
        shortcuts_group = QGroupBox("Keyboard Shortcuts")
        shortcuts_layout = QVBoxLayout(shortcuts_group)
        shortcuts_text = """
M - Toggle Microphone
V - Toggle Camera
S - Toggle Screen Share
H - Raise/Lower Hand
C - Toggle Chat Panel
F11 - Toggle Fullscreen
        """
        shortcuts_label = QLabel(shortcuts_text.strip())
        shortcuts_label.setStyleSheet("color: #888; font-family: monospace;")
        shortcuts_layout.addWidget(shortcuts_label)
        layout.addWidget(shortcuts_group)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main client entry point."""
    if len(sys.argv) < 2:
        print("Usage: python client.py <config.yaml>")
        sys.exit(1)
   
    config_file = sys.argv[1]
   
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
   
    logger.info("Client started")
   
    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
   
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
