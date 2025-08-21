# Criminal Face Detection System

A real-time criminal face detection and monitoring system designed for security enforcement applications.

## Overview

This system provides real-time monitoring capabilities to detect known criminals through camera feeds. It features a clean, professional interface with high-performance optimization for reliable security operations.

## Features

- **Real-time Face Detection**: Advanced face recognition using InsightFace
- **Criminal Database**: Easy management of criminal profiles
- **High Performance**: Optimized for 15-25 FPS real-time processing with quantization support
- **Professional UI**: Clean red-themed interface for security operations
- **Multiple Camera Support**:
  - **Single Camera Mode**: Traditional single camera monitoring (backward compatible)
  - **Multi-Camera Mode**: Simultaneous monitoring of multiple cameras
  - **Auto Camera Detection**: Automatically detects available cameras
  - **Flexible Layout**: Adaptive UI layout based on number of cameras
- **Instant Alerts**: Immediate notifications when criminals are detected across all cameras
- **Performance Optimization**: Model quantization for 40-50% performance boost
- **Smart Threshold Adjustment**: Automatic threshold compensation for quantization

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
.\env\Scripts\Activate.ps1

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Add Criminal Profiles

- Place criminal photos in the `face_db/` folder
- Or use the web interface to upload new profiles

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Access the Interface

Open your browser to `http://localhost:8501`

## Usage

### Adding Criminal Profiles

1. Use the sidebar "Add New Criminal Profile" section
2. Upload an image and enter the name/ID
3. Click "Add Profile" to save to database

### Camera Setup

1. **Single Camera Mode** (Default for single camera setups):

   - Select your camera from the dropdown
   - Perfect for laptop webcams or single USB cameras
   - Maintains full backward compatibility

2. **Multi-Camera Mode** (For multiple camera setups):
   - Automatically detects all available cameras
   - Select multiple cameras for simultaneous monitoring
   - Adaptive grid layout (2x2, 3x3, etc.)
   - Independent performance metrics for each camera

### Monitoring

1. **Camera Selection**:
   - Choose between "Single Camera" or "Multiple Cameras" mode
   - View detected cameras with resolution and FPS information
2. **Performance Settings**: Enable quantization for enhanced performance
3. **Detection Settings**: Adjust confidence and threshold settings
4. **Start Monitoring**: Begin real-time detection across selected cameras
5. **Monitor Alerts**: Watch for criminal detection notifications from any camera

### Settings

- **Camera Mode**: Choose single or multiple camera monitoring
- **Detection Confidence**: Adjust face detection sensitivity
- **Face Match Threshold**: Control criminal recognition accuracy
- **Performance Settings**:
  - **Model Quantization**: Enable for 40-50% performance boost
  - **Max FPS**: Optimize frame rate
  - **Detection Size**: Choose processing resolution for performance
- **Multi-Camera Layout**: Automatic grid arrangement for multiple cameras

## Technical Specifications

- **Framework**: Streamlit + OpenCV + InsightFace + ONNX Runtime
- **Performance**: 15-25 FPS real-time processing (up to 35 FPS with quantization)
- **Detection Models**: CPU-optimized face recognition with quantization support
- **Supported Formats**: PNG, JPG, JPEG for criminal profiles
- **Camera Support**:
  - USB cameras, webcams, IP cameras
  - Single camera mode: Full backward compatibility
  - Multi-camera mode: Simultaneous monitoring up to 5 cameras
  - Automatic camera detection and configuration
- **Multi-Threading**: Concurrent frame capture and processing
- **Memory Optimization**: Efficient resource management for multiple streams

## Project Structure

```
face_detection/
├── app.py                 # Main application
├── face_db/              # Criminal profiles database
│   ├── criminal1.jpg
│   └── criminal2.jpg
├── env/                  # Python virtual environment
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Security Notes

- This system is designed for legitimate security enforcement use
- Ensure proper authorization before deploying for surveillance
- Criminal database should be maintained by authorized personnel only
- Follow local privacy and surveillance regulations

## Performance Optimization

The system is optimized for maximum performance:

- Ultra-fast detection mode (128x128 processing)
- Frame skipping for high FPS
- Vectorized similarity computations
- Detection persistence to reduce false negatives
- Minimal memory footprint

## Support

For technical support or questions about deployment, refer to the application's built-in help or contact the development team.

---

_Criminal Face Detection System - Professional Security Solution_
