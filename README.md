# 🎥 LAN Video Conferencing Application

A professional-grade, LAN-based video conferencing application built with Python and PyQt6. Supports multi-user video/audio conferencing, screen sharing, chat, file sharing, and many modern collaboration features.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-6.4+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### Core Features
- **🎥 Video Conferencing** - Multi-user video with dynamic grid layout
- **🎤 Audio Conferencing** - Real-time audio with noise suppression
- **🖥️ Screen Sharing** - Share your screen with all participants
- **💬 Chat** - Group and private messaging
- **📁 File Sharing** - Upload and download files during meetings

### Advanced Features
- **✋ Raise Hand** - Signal to speak without interrupting
- **😀 Reactions** - Quick emoji reactions (👍 ❤️ 👏 😂 😮 🎉 🔥 💯)
- **⏺️ Recording** - Record meetings to MP4
- **🌈 Virtual Backgrounds** - Blur or custom image backgrounds
- **🔇 Noise Suppression** - Filter out background noise
- **📌 Pin Video** - Pin any participant's video
- **⌨️ Keyboard Shortcuts** - Quick access to all controls

### User Interface
- **🌙 Dark Theme** - Modern, eye-friendly dark interface
- **📊 Dynamic Grid** - Auto-adjusting video grid layout
- **🖼️ Fullscreen Mode** - Immersive fullscreen support
- **⚙️ Settings Panel** - Customizable preferences

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Camera and microphone (optional)
- Same LAN network for all participants

### Installation

```bash
# Clone the repository
git clone https://github.com/SriSaiKrishna18/LAN-Based-Video-Conferencing-Application.git
cd LAN-Based-Video-Conferencing-Application

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**1. Start the Server** (on one machine):
```bash
python server.py config.yaml
```

**2. Start the Client** (on each participant's machine):
```bash
python client.py config.yaml
```

> **Note**: Update `config.yaml` with the server's IP address before connecting clients.

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `M` | Toggle Microphone |
| `V` | Toggle Camera |
| `S` | Toggle Screen Share |
| `H` | Raise/Lower Hand |
| `C` | Toggle Chat Panel |
| `F11` | Toggle Fullscreen |

## ⚙️ Configuration

Edit `config.yaml` to customize settings:

```yaml
server:
  host: "192.168.1.100"  # Server IP address
  control_tcp_port: 5000
  file_tcp_port: 5001
  screen_tcp_port: 5002

media:
  video_quality: "medium"  # low, medium, high
  audio_sample_rate: 16000

features:
  enable_reactions: true
  enable_recording: true
  enable_virtual_background: true
```

## 📁 Project Structure

```
├── client.py          # Client application (GUI + media)
├── server.py          # Server application (signaling + routing)
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── recordings/        # Saved meeting recordings
├── uploads/           # Shared files
└── backgrounds/       # Custom virtual backgrounds
```

## 🛠️ Technology Stack

- **GUI**: PyQt6
- **Video Processing**: OpenCV, PyAV
- **Audio**: PyAudio
- **Screen Capture**: MSS
- **Networking**: TCP/UDP Sockets
- **Audio Processing**: SciPy (noise suppression)

## 📋 Requirements

- Python 3.9+
- PyQt6 >= 6.4.0
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- av >= 10.0.0
- pyaudio >= 0.2.12
- mss >= 7.0.0
- PyYAML >= 6.0
- scipy >= 1.9.0 (optional, for noise suppression)

## 🔧 Troubleshooting

### Camera/Microphone not working
- Ensure devices are not in use by other applications
- Check device permissions in system settings
- Try changing device indices in `config.yaml`

### Connection issues
- Verify server IP address in `config.yaml`
- Check firewall settings (allow ports 5000-5006)
- Ensure all machines are on the same LAN

### Video quality issues
- Lower video quality in settings if experiencing lag
- Check network bandwidth

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyQt6 for the excellent GUI framework
- OpenCV and PyAV for media processing
- The Python community for amazing libraries

---

Made with ❤️ for LAN-based collaboration
