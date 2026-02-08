"""
LAN Collab Server - Standalone Version
Multi-user video/audio conferencing, chat, file sharing, and screen sharing over LAN.
"""

import sys
import logging
import socket
import threading
import json
import struct
import uuid
import signal
from pathlib import Path
from typing import Dict, List, Optional

import yaml

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
# USER REGISTRY
# ============================================================================

class Registry:
    """Thread-safe user registry."""
    
    def __init__(self):
        self._users: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def add_user(self, user_id: str, username: str, sock, address: tuple):
        with self._lock:
            self._users[user_id] = {
                "user_id": user_id,
                "username": username,
                "socket": sock,
                "address": address,
            }
            logger.info(f"User added: {username} ({user_id})")
    
    def remove_user(self, user_id: str):
        with self._lock:
            if user_id in self._users:
                uname = self._users[user_id]["username"]
                del self._users[user_id]
                logger.info(f"User removed: {uname} ({user_id})")
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        with self._lock:
            return self._users.get(user_id)
    
    def get_all_users(self) -> List[Dict]:
        with self._lock:
            return list(self._users.values())
    
    def find_by_username(self, username: str) -> Optional[Dict]:
        with self._lock:
            for u in self._users.values():
                if u["username"] == username:
                    return u
            return None
    
    def count(self) -> int:
        with self._lock:
            return len(self._users)


# ============================================================================
# CONTROL SERVER
# ============================================================================

class ControlServer:
    """TCP-based control server for chat and signaling."""
    
    def __init__(self, config: dict, registry: Registry):
        self.config = config
        self.registry = registry
        self.host = config["server"]["bind_host"]
        self.port = config["server"]["control_tcp_port"]
        self._srv = None
        self._running = False
    
    def start(self):
        """Start control server."""
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self.port))
        self._srv.listen(64)
        self._running = True
        logger.info(f"Control server listening on {self.host}:{self.port}")
        threading.Thread(target=self._accept_loop, daemon=True).start()
    
    def stop(self):
        """Stop control server."""
        self._running = False
        if self._srv:
            try:
                self._srv.close()
            except:
                pass
    
    def _accept_loop(self):
        """Accept client connections."""
        while self._running:
            try:
                cli, addr = self._srv.accept()
                logger.info(f"Control connection from {addr}")
                threading.Thread(target=self._client_loop, args=(cli, addr), daemon=True).start()
            except Exception as e:
                if self._running:
                    logger.error(f"Accept error: {e}")
    
    def _client_loop(self, sock: socket.socket, addr: tuple):
        """Handle client connection."""
        user_id = None
        username = None
        try:
            while self._running:
                raw_len = self._recv_exact(sock, 4)
                if not raw_len:
                    break
                n = int.from_bytes(raw_len, "big")
                raw_msg = self._recv_exact(sock, n)
                if not raw_msg:
                    break
                msg = json.loads(raw_msg.decode("utf-8"))
                mtype = msg.get("type")
                
                if mtype == "join":
                    username = msg.get("username", "Unknown")
                    user_id = str(uuid.uuid4())
                    self.registry.add_user(user_id, username, sock, addr)
                    self._send(sock, {"type": "join_ack", "user_id": user_id, "username": username})
                    self._broadcast_roster()
                    logger.info(f"User joined: {username} ({user_id})")
                
                elif mtype == "chat" and user_id:
                    text = msg.get("text", "")
                    recipient = msg.get("to")
                    out = {"type": "chat", "from": username, "text": text, "to": recipient if recipient else "Everyone"}
                    
                    if recipient:
                        target = self.registry.find_by_username(recipient)
                        if target:
                            try:
                                self._send(target["socket"], out)
                                self._send(sock, out)
                            except:
                                pass
                            logger.info(f"Private message {username} -> {recipient}")
                    else:
                        self._broadcast_except(user_id, out)
                
                elif mtype == "reaction":
                    # Broadcast emoji reaction to all participants
                    emoji = msg.get("emoji", "")
                    reaction_msg = {
                        "type": "reaction",
                        "username": username,
                        "emoji": emoji
                    }
                    self._broadcast_except(user_id, reaction_msg)
                    logger.info(f"Reaction from {username}: {emoji}")
                
                elif mtype == "hand_raise":
                    # Broadcast hand raise/lower to all participants
                    raised = msg.get("raised", False)
                    hand_msg = {
                        "type": "hand_raise",
                        "username": username,
                        "raised": raised
                    }
                    self._broadcast_except(user_id, hand_msg)
                    logger.info(f"Hand {'raised' if raised else 'lowered'} by {username}")
                
                elif mtype == "leave":
                    break
        
        except Exception as e:
            logger.error(f"Client loop error {addr}: {e}")
        finally:
            if user_id:
                self.registry.remove_user(user_id)
                self._broadcast_roster()
            try:
                sock.close()
            except:
                pass
            logger.info(f"Client {addr} disconnected")
    
    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes."""
        buf = b""
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            except:
                return None
        return buf
    
    def _send(self, sock: socket.socket, obj: dict):
        """Send JSON message."""
        data = json.dumps(obj).encode("utf-8")
        sock.sendall(len(data).to_bytes(4, "big") + data)
    
    def _broadcast_roster(self):
        """Broadcast user roster to all clients."""
        users = self.registry.get_all_users()
        payload = {"type": "roster", "users": [{"username": u["username"]} for u in users]}
        for u in users:
            try:
                self._send(u["socket"], payload)
            except:
                pass
    
    def _broadcast_except(self, sender_id: str, obj: dict):
        """Broadcast message to all except sender."""
        users = self.registry.get_all_users()
        for u in users:
            if u["user_id"] == sender_id:
                continue
            try:
                self._send(u["socket"], obj)
            except:
                pass


# ============================================================================
# FILE SERVER
# ============================================================================

class FileServer:
    """TCP-based file server."""
    
    def __init__(self, config: dict, registry: Registry):
        self.config = config
        self.registry = registry
        self.host = config["server"]["bind_host"]
        self.port = config["server"]["file_tcp_port"]
        self.socket = None
        self.running = False
        self.thread = None
        self.files = {}
        upload_dir_name = config["server"].get("upload_dir", "uploads")
        self.upload_dir = Path(upload_dir_name)
        self.upload_dir.mkdir(exist_ok=True)
    
    def start(self):
        """Start file server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            
            self.running = True
            self.thread = threading.Thread(target=self._accept_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"File server listening on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start file server: {e}")
            return False
    
    def stop(self):
        """Stop file server."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        logger.info("File server stopped")
    
    def _accept_loop(self):
        """Accept client connections."""
        while self.running:
            try:
                client_sock, addr = self.socket.accept()
                logger.info(f"File connection from {addr}")
                handler = threading.Thread(target=self._handle_client, args=(client_sock, addr), daemon=True)
                handler.start()
            except:
                if self.running:
                    logger.error("Socket error in file server")
                break
    
    def _handle_client(self, client_sock: socket.socket, addr):
        """Handle file client."""
        try:
            client_sock.settimeout(30.0)
            
            data = b""
            while b"\n" not in data:
                chunk = client_sock.recv(4096)
                if not chunk:
                    return
                data += chunk
                if len(data) > 1000000:
                    return
            
            request_line, remaining = data.split(b"\n", 1)
            request = json.loads(request_line.decode())
            action = request.get("action")
            
            if action == "upload":
                self._handle_upload(client_sock, request, remaining, addr)
            elif action == "download":
                self._handle_download(client_sock, request, addr)
            elif action == "list":
                self._handle_list(client_sock, addr)
        
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}")
        finally:
            try:
                client_sock.close()
            except:
                pass
    
    def _handle_upload(self, sock: socket.socket, request: dict, initial_data: bytes, addr):
        """Handle file upload."""
        try:
            filename = request.get("filename", "unknown")
            size = request.get("size", 0)
            
            logger.info(f"Upload request: {filename} ({size} bytes) from {addr}")
            
            file_id = str(uuid.uuid4())
            response = {"status": "ok", "file_id": file_id}
            sock.sendall((json.dumps(response) + "\n").encode())
            
            received = len(initial_data)
            file_data = bytearray(initial_data)
            
            while received < size:
                chunk = sock.recv(min(65536, size - received))
                if not chunk:
                    return
                file_data.extend(chunk)
                received += len(chunk)
            
            save_path = self.upload_dir / f"{file_id}_{filename}"
            save_path.write_bytes(file_data)
            
            self.files[file_id] = {
                "filename": filename,
                "size": size,
                "path": str(save_path),
                "uploader": addr[0]
            }
            
            logger.info(f"Upload complete: {filename} -> {file_id}")
        except Exception as e:
            logger.error(f"Upload error: {e}")
    
    def _handle_download(self, sock: socket.socket, request: dict, addr):
        """Handle file download."""
        try:
            file_id = request.get("file_id")
            
            if not file_id or file_id not in self.files:
                response = {"error": "file not found"}
                sock.sendall((json.dumps(response) + "\n").encode())
                return
            
            meta = self.files[file_id]
            file_path = Path(meta["path"])
            
            if not file_path.exists():
                response = {"error": "file deleted"}
                sock.sendall((json.dumps(response) + "\n").encode())
                return
            
            response = {
                "status": "ok",
                "filename": meta["filename"],
                "size": meta["size"]
            }
            sock.sendall((json.dumps(response) + "\n").encode())
            
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    sock.sendall(chunk)
            
            logger.info(f"Download complete: {meta['filename']} to {addr}")
        except Exception as e:
            logger.error(f"Download error: {e}")
    
    def _handle_list(self, sock: socket.socket, addr):
        """Send list of available files."""
        try:
            files_list = [
                {
                    "file_id": fid,
                    "filename": meta["filename"],
                    "size": meta["size"]
                }
                for fid, meta in self.files.items()
            ]
            
            response = {"files": files_list}
            sock.sendall((json.dumps(response) + "\n").encode())
        except Exception as e:
            logger.error(f"List error: {e}")


# ============================================================================
# VIDEO ROUTER
# ============================================================================

class VideoRouter:
    """UDP-based video packet router."""
    
    def __init__(self, config: dict, registry: Registry):
        self.config = config
        self.registry = registry
        self.host = config["server"]["bind_host"]
        self.port = config["media"]["video_rtp_port"]
        self.socket = None
        self.running = False
        self.thread = None
        self.receivers = []
        self.senders = {}
        self.lock = threading.Lock()
    
    def start(self):
        """Start video router."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.host, self.port))
            
            self.running = True
            self.thread = threading.Thread(target=self._route_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"Video router listening on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start video router: {e}")
            return False
    
    def stop(self):
        """Stop video router."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        logger.info("Video router stopped")
    
    def _route_loop(self):
        """Route video packets."""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536)
                
                if data == b'VIDEO_RECEIVER_REGISTER':
                    with self.lock:
                        if addr not in self.receivers:
                            self.receivers.append(addr)
                            logger.info(f"Registered video receiver at {addr}")
                    continue
                
                if len(data) < 12:
                    continue
                
                try:
                    payload, ssrc, seq, timestamp, pt, marker = unpack_rtp(data)
                except:
                    continue
                
                with self.lock:
                    # Track sender by SSRC, not just address
                    if ssrc not in self.senders:
                        self.senders[ssrc] = addr
                        logger.info(f"Registered video sender SSRC {ssrc} from {addr}")
                    
                    # Get the sender's address for this SSRC
                    sender_addr = self.senders.get(ssrc)
                    
                    # Broadcast to ALL receivers including sender
                    # This allows users to see their own video
                    for receiver_addr in self.receivers:
                        try:
                            self.socket.sendto(data, receiver_addr)
                        except:
                            pass
            except:
                if self.running:
                    continue
                else:
                    break


# ============================================================================
# AUDIO MIXER
# ============================================================================

class AudioMixer:
    """UDP-based audio mixer with echo suppression."""
    
    def __init__(self, config: dict, registry: Registry):
        self.config = config
        self.registry = registry
        self.host = config["server"]["bind_host"]
        self.port = config["media"]["audio_rtp_port"]
        self.socket = None
        self.running = False
        self.clients = {}
        self.senders = {}
        self.lock = threading.Lock()
    
    def start(self):
        """Start audio mixer."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.host, self.port))
            self.running = True
            logger.info(f"Audio mixer listening on {self.host}:{self.port}")
            threading.Thread(target=self._receive_loop, daemon=True).start()
            return True
        except Exception as e:
            logger.error(f"Failed to start audio mixer: {e}")
            return False
    
    def stop(self):
        """Stop audio mixer."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        logger.info("Audio mixer stopped")
    
    def _receive_loop(self):
        """Receive and forward audio packets."""
        while self.running:
            try:
                data, sender_addr = self.socket.recvfrom(2048)
                
                if data == b'AUDIO_RECEIVER_REGISTER':
                    with self.lock:
                        self.clients[sender_addr] = True
                    logger.info(f"Registered audio client at {sender_addr}")
                    continue
                
                if len(data) < 12:
                    continue
                
                try:
                    payload, ssrc, seq, ts, pt, marker = unpack_rtp(data)
                except:
                    continue
                
                with self.lock:
                    if ssrc not in self.senders:
                        self.senders[ssrc] = sender_addr
                        logger.info(f"Registered audio sender SSRC {ssrc} from {sender_addr}")
                    
                    for client_addr in list(self.clients.keys()):
                        if client_addr == sender_addr:
                            continue
                        
                        try:
                            self.socket.sendto(data, client_addr)
                        except:
                            self.clients.pop(client_addr, None)
            except:
                if self.running:
                    continue
                else:
                    break


# ============================================================================
# SCREEN SERVER
# ============================================================================

class ScreenServer:
    """TCP-based screen sharing relay."""
    
    def __init__(self, config: dict, registry: Registry):
        self.config = config
        self.registry = registry
        self.host = config["server"]["bind_host"]
        self.port = config["server"]["screen_tcp_port"]
        self.presenter_socket = None
        self.presenter_lock = threading.Lock()
        self.viewers = []
        self.viewers_lock = threading.Lock()
        self.running = False
        self.server_socket = None
    
    def start(self):
        """Start screen server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(32)
            self.running = True
            
            logger.info(f"Screen server listening on {self.host}:{self.port}")
            
            accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
            accept_thread.start()
        except Exception as e:
            logger.error(f"Failed to start screen server: {e}")
            raise
    
    def stop(self):
        """Stop screen server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
    
    def _accept_loop(self):
        """Accept incoming connections."""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"Screen connection from {addr}")
                
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, addr),
                    daemon=True
                )
                client_thread.start()
            except:
                if self.running:
                    logger.error("Error accepting screen connection")
    
    def _handle_client(self, client_socket: socket.socket, addr):
        """Handle screen share connection."""
        try:
            client_socket.settimeout(10.0)
            
            msg = read_msg(client_socket)
            if not msg:
                client_socket.close()
                return
            
            role = msg.get("role")
            
            if role == "presenter":
                self._handle_presenter(client_socket, addr)
            elif role == "viewer":
                self._handle_viewer(client_socket, addr)
        except:
            pass
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def _handle_presenter(self, client_socket: socket.socket, addr):
        """Handle presenter connection."""
        with self.presenter_lock:
            if self.presenter_socket:
                write_msg(client_socket, {"status": "error", "message": "Presenter already active"})
                client_socket.close()
                return
            
            self.presenter_socket = client_socket
            write_msg(client_socket, {"status": "ok"})
        
        logger.info(f"Presenter started from {addr}")
        
        try:
            client_socket.settimeout(None)
            
            while self.running:
                frame_data = read_msg(client_socket)
                if not frame_data:
                    break
                
                with self.viewers_lock:
                    dead_viewers = []
                    for i, viewer_socket in enumerate(self.viewers):
                        if not write_msg(viewer_socket, frame_data):
                            dead_viewers.append(i)
                    
                    for i in reversed(dead_viewers):
                        try:
                            self.viewers[i].close()
                        except:
                            pass
                        self.viewers.pop(i)
        except:
            pass
        finally:
            with self.presenter_lock:
                self.presenter_socket = None
            logger.info(f"Presenter {addr} stopped")
    
    def _handle_viewer(self, client_socket: socket.socket, addr):
        """Handle viewer connection."""
        with self.viewers_lock:
            self.viewers.append(client_socket)
            write_msg(client_socket, {"status": "ok"})
        
        logger.info(f"Screen viewer added from {addr}")
        
        try:
            client_socket.settimeout(None)
            while self.running:
                data = client_socket.recv(1)
                if not data:
                    break
        except:
            pass
        finally:
            with self.viewers_lock:
                try:
                    self.viewers.remove(client_socket)
                except:
                    pass
            logger.info(f"Viewer {addr} removed")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main server entry point."""
    if len(sys.argv) < 2:
        print("Usage: python server.py <config.yaml>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_file}")
    
    upload_dir = Path(config.get("server", {}).get("upload_dir", "uploads"))
    upload_dir.mkdir(exist_ok=True)
    
    registry = Registry()
    
    control_server = ControlServer(config, registry)
    file_server = FileServer(config, registry)
    video_router = VideoRouter(config, registry)
    audio_mixer = AudioMixer(config, registry)
    screen_server = ScreenServer(config, registry)
    
    try:
        control_server.start()
        file_server.start()
        video_router.start()
        audio_mixer.start()
        screen_server.start()
        
        logger.info("Server started successfully")
        
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
        
        while True:
            threading.Event().wait(1)
    
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        control_server.stop()
        file_server.stop()
        video_router.stop()
        audio_mixer.stop()
        screen_server.stop()
    
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()