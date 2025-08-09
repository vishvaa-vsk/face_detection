# Face Detection and Matching

A simple face detection and real-time face matching project with two entry points:
- `app.py`: Streamlit web app for webcam face matching with a small face database.
- `newnewtest.py`: Minimal OpenCV window app doing the same in a plain desktop UI.

## Features
- Detect faces and compute embeddings using InsightFace.
- Compare live embeddings against images in a local folder (`face_db/`).
- Threshold-based match labeling: red box for match, green for unknown.
- Upload new faces to the DB via Streamlit sidebar.

## Prerequisites
- Python 3.9–3.12 recommended.
- On Linux, ensure system packages for OpenCV video I/O are present (e.g., `ffmpeg`, `v4l2`, `libgl1`).
- A working webcam for live mode.

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you run into onnxruntime or OpenCV import errors on Linux, install:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 ffmpeg v4l2loopback-dkms
```

## Usage

Create a `face_db/` folder with one image per person. Filenames (without extension) are used as labels.

- Run the Streamlit app:

```bash
streamlit run app.py
```

- Run the OpenCV script:

```bash
python newnewtest.py
```

Press `q` in the OpenCV window to quit.

## Notes
- Threshold defaults to 0.35 and is adjustable in the Streamlit app. Tune per your dataset.
- CPU backend is used via `onnxruntime` (`CPUExecutionProvider`). GPU would require installing `onnxruntime-gpu` and proper CUDA drivers.
- For best results, use clear, frontal face images in the database.
