# Criminal Face Detection System

A real-time criminal face detection and monitoring system designed for security enforcement applications.

## Overview

This system provides real-time monitoring capabilities to detect known criminals through camera feeds. It features a clean, professional interface with high-performance optimization for reliable security operations.

## Features

- **Real-time Face Detection**: Advanced face recognition using InsightFace
- **Criminal Database**: Easy management of criminal profiles
- **High Performance**: Optimized for 15-25 FPS real-time processing
- **Professional UI**: Clean red-themed interface for security operations
- **Multiple Camera Support**: Support for multiple camera inputs
- **Instant Alerts**: Immediate notifications when criminals are detected

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

### Monitoring

1. Select your camera from the dropdown
2. Adjust detection settings as needed
3. Click "Start Monitoring" to begin real-time detection
4. Monitor alerts for criminal detection notifications

### Settings

- **Detection Confidence**: Adjust face detection sensitivity
- **Face Match Threshold**: Control criminal recognition accuracy
- **Performance Settings**: Optimize speed vs accuracy
- **Detection Size**: Choose processing resolution for performance

## Technical Specifications

- **Framework**: Streamlit + OpenCV + InsightFace
- **Performance**: 15-25 FPS real-time processing
- **Detection Models**: CPU-optimized face recognition
- **Supported Formats**: PNG, JPG, JPEG for criminal profiles
- **Camera Support**: USB cameras, webcams, IP cameras

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
