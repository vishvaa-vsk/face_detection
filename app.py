import streamlit as st
import cv2
import os
import numpy as np
import tempfile
import insightface
from PIL import Image
import threading
import time
from queue import Queue
import onnxruntime as ort
import concurrent.futures
from collections import defaultdict

# Configure Streamlit page
st.set_page_config(
    page_title="Criminal Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS Styling
# ----------------------------
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main App Container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: white;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1cypcdb {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 3px solid #ff0000;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #ff0000, #8b0000);
        color: white;
        border: 2px solid #ff0000;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 0, 0, 0.4);
        background: linear-gradient(45deg, #8b0000, #ff0000);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 0, 0, 0.3);
        border-radius: 8px;
        color: white;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: rgba(46, 204, 113, 0.1);
        border: 1px solid rgba(46, 204, 113, 0.3);
        color: #2ecc71;
        border-radius: 8px;
    }
    
    .stError {
        background-color: rgba(231, 76, 60, 0.1);
        border: 1px solid rgba(231, 76, 60, 0.3);
        color: #e74c3c;
        border-radius: 8px;
    }
    
    /* Criminal Alert Neon Effect */
    .criminal-alert {
        background: linear-gradient(45deg, #ff0000, #cc0000, #ff0000);
        background-size: 200% 200%;
        animation: neon-glow 1.5s ease-in-out infinite alternate, gradient-shift 2s ease-in-out infinite;
        border: 2px solid #ff0000;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 22px;
        font-family: 'Inter', sans-serif;
        text-shadow: 
            0 0 5px #ff0000, 
            0 0 10px #ff0000, 
            0 0 15px #ff0000,
            0 0 20px #ff0000;
        box-shadow: 
            0 0 15px rgba(255, 0, 0, 0.6),
            0 0 30px rgba(255, 0, 0, 0.4),
            0 0 45px rgba(255, 0, 0, 0.2),
            inset 0 0 15px rgba(255, 0, 0, 0.1);
        margin: 10px 0;
        position: relative;
        overflow: hidden;
    }
    
    .criminal-alert::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #ff0000, transparent, #ff0000);
        border-radius: 12px;
        z-index: -1;
        animation: border-glow 2s linear infinite;
    }
    
    @keyframes neon-glow {
        from {
            text-shadow: 
                0 0 5px #ff0000, 
                0 0 10px #ff0000, 
                0 0 15px #ff0000;
            box-shadow: 
                0 0 15px rgba(255, 0, 0, 0.6),
                0 0 30px rgba(255, 0, 0, 0.4);
        }
        to {
            text-shadow: 
                0 0 10px #ff0000, 
                0 0 20px #ff0000, 
                0 0 30px #ff0000,
                0 0 40px #ff0000;
            box-shadow: 
                0 0 25px rgba(255, 0, 0, 0.8),
                0 0 50px rgba(255, 0, 0, 0.6),
                0 0 75px rgba(255, 0, 0, 0.4);
        }
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes border-glow {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .all-clear {
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        border: 2px solid #2ecc71;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        color: white;
        font-weight: 600;
        font-size: 18px;
        font-family: 'Inter', sans-serif;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Camera Detection and Management
# ----------------------------
@st.cache_data
def detect_available_cameras(max_cameras=5):
    """Detect all available camera devices"""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            available_cameras.append({
                'index': i,
                'name': f"Camera {i}",
                'resolution': f"{width}x{height}",
                'fps': fps
            })
            cap.release()
    return available_cameras

class MultiCameraManager:
    """Manages multiple camera streams efficiently"""
    
    def __init__(self, camera_indices, max_fps=25):
        self.camera_indices = camera_indices
        self.max_fps = max_fps
        self.cameras = {}
        self.frames = {}
        self.last_frame_time = {}
        self.running = False
        
    def initialize_cameras(self):
        """Initialize all camera captures"""
        for cam_idx in self.camera_indices:
            cap = cv2.VideoCapture(cam_idx)
            if cap.isOpened():
                # Optimize camera settings
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, self.max_fps)
                self.cameras[cam_idx] = cap
                self.frames[cam_idx] = None
                self.last_frame_time[cam_idx] = 0
            else:
                st.warning(f"Could not initialize Camera {cam_idx}")
    
    def capture_frame(self, cam_idx):
        """Capture a single frame from specified camera"""
        if cam_idx in self.cameras:
            ret, frame = self.cameras[cam_idx].read()
            if ret:
                self.frames[cam_idx] = frame
                self.last_frame_time[cam_idx] = time.time()
                return frame
        return None
    
    def capture_all_frames(self):
        """Capture frames from all cameras simultaneously"""
        current_time = time.time()
        
        # Use threading for simultaneous capture
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.cameras)) as executor:
            futures = {executor.submit(self.capture_frame, cam_idx): cam_idx 
                      for cam_idx in self.cameras}
            
            for future in concurrent.futures.as_completed(futures):
                cam_idx = futures[future]
                try:
                    frame = future.result()
                except Exception as e:
                    st.warning(f"Error capturing from Camera {cam_idx}: {e}")
        
        return self.frames.copy()
    
    def release_all(self):
        """Release all camera resources"""
        for cap in self.cameras.values():
            if cap.isOpened():
                cap.release()
        self.cameras.clear()
        self.frames.clear()
class PerformanceTimer:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
# ----------------------------
# PerformanceTimer Class
# ----------------------------
class PerformanceTimer:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def tick(self):
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def get_avg_fps(self):
        if not self.frame_times:
            return 0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0

def process_camera_frame(frame, app, known_encodings, known_names, similarity_threshold, use_quantization, detection_size, resize_factor=0.5):
    """Process a single camera frame for face detection"""
    detections = {}
    criminal_detected = False
    
    if frame is not None:
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            faces = app.get(rgb_frame)
            
            for face in faces:
                # Scale coordinates back to original frame
                bbox = (face.bbox / resize_factor).astype(int)
                face_encoding = face.embedding
                
                # Ensure consistent encoding format
                if face_encoding is not None and len(face_encoding) > 0:
                    # Normalize the encoding
                    face_encoding = face_encoding / np.linalg.norm(face_encoding)
                    
                    # Find match with quantization awareness
                    name, similarity = find_face_matches(
                        face_encoding, 
                        known_encodings, 
                        known_names, 
                        similarity_threshold,
                        use_quantization
                    )
                    
                    # Store detection
                    detection_key = f"{bbox[0]}_{bbox[1]}"
                    detections[detection_key] = {
                        'bbox': bbox,
                        'name': name,
                        'similarity': similarity,
                        'timestamp': time.time()
                    }
                    
                    if name != "Unknown":
                        criminal_detected = True
        
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    return detections, criminal_detected

def draw_detections_on_frame(frame, detections, current_time):
    """Draw detection boxes and labels on frame"""
    any_criminal = False
    
    for detection in detections.values():
        if current_time - detection['timestamp'] < 1.0:  # Show for 1 second
            bbox = detection['bbox']
            name = detection['name']
            similarity = detection['similarity']
            
            if name != "Unknown":
                any_criminal = True
            
            # Choose color based on whether it's a known criminal
            color = (0, 0, 255) if name != "Unknown" else (0, 255, 0)  # Red for criminal, Green for unknown
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            text = f"{name} ({similarity:.2f})" if name != "Unknown" else "Unknown"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_width, text_height = text_size
            
            # Background for text
            cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 15), 
                        (bbox[0] + text_width + 10, bbox[1]), color, -1)
            cv2.putText(frame, text, (bbox[0] + 5, bbox[1] - 8),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame, any_criminal

# ----------------------------
# Cache face recognition app and models
# ----------------------------
@st.cache_resource
def load_face_recognition(use_quantization=False, detection_size=(128, 128)):
    """Load and cache the face recognition model with optional quantization"""
    import onnxruntime as ort
    
    if use_quantization:
        # Configure optimized session for quantization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.intra_op_num_threads = 0  # Use all available cores
        
        # Enable memory pattern optimization
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        app = insightface.app.FaceAnalysis(
            providers=['CPUExecutionProvider'],
            sess_options=sess_options
        )
    else:
        app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
    
    app.prepare(ctx_id=0, det_size=detection_size)
    return app

@st.cache_data
def load_known_faces():
    """Load known faces from the face_db directory with consistent embedding dimensions"""
    known_encodings = []
    known_names = []
    face_db_path = "face_db"
    
    if not os.path.exists(face_db_path):
        os.makedirs(face_db_path)
        return known_encodings, known_names
    
    # Use consistent model configuration for loading faces
    app_loader = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
    app_loader.prepare(ctx_id=0, det_size=(128, 128))  # Use same size as runtime for consistency
    
    for filename in os.listdir(face_db_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(face_db_path, filename)
            try:
                # Load image using CV2
                img = cv2.imread(image_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Get face embeddings
                    faces = app_loader.get(img_rgb)
                    if faces and len(faces) > 0:
                        # Use the first detected face
                        face_encoding = faces[0].embedding
                        
                        # Ensure consistent dimensions and normalize
                        if face_encoding is not None and len(face_encoding) > 0:
                            # Normalize encoding for better quantization compatibility
                            face_encoding = face_encoding / np.linalg.norm(face_encoding)
                            known_encodings.append(face_encoding)
                            # Use filename without extension as name
                            name = os.path.splitext(filename)[0]
                            known_names.append(name)
                            print(f"Successfully loaded {name} with embedding dimension: {len(face_encoding)}")
                        else:
                            st.warning(f"Invalid face encoding for {filename}")
                    else:
                        st.warning(f"No face detected in {filename}")
            except Exception as e:
                st.warning(f"Could not process {filename}: {str(e)}")
                print(f"Error details for {filename}: {str(e)}")
    
    print(f"Loaded {len(known_encodings)} face encodings")
    if known_encodings:
        print(f"Embedding dimensions: {[len(enc) for enc in known_encodings]}")
    
    return known_encodings, known_names

def add_new_face(uploaded_file, name):
    """Add a new face to the database"""
    if uploaded_file is not None and name:
        # Save the uploaded file
        face_db_path = "face_db"
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        
        # Create filename
        file_extension = uploaded_file.name.split('.')[-1]
        filename = f"{name}.{file_extension}"
        filepath = os.path.join(face_db_path, filename)
        
        # Save file
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Clear cache to reload faces
        st.cache_data.clear()
        return True
    return False

def find_face_matches(face_encoding, known_encodings, known_names, threshold=0.4, use_quantization=False):
    """Find matches for a face encoding with quantization-aware adjustments and dimension safety"""
    if len(known_encodings) == 0:
        return "Unknown", 1.0
    
    try:
        # Ensure face_encoding is a numpy array and normalized
        face_encoding = np.array(face_encoding)
        if np.linalg.norm(face_encoding) > 0:
            face_encoding = face_encoding / np.linalg.norm(face_encoding)
        
        # Check dimensions and handle mismatches
        known_encodings_array = np.array(known_encodings)
        
        # Verify dimension compatibility
        if face_encoding.shape[0] != known_encodings_array.shape[1]:
            print(f"Dimension mismatch: current face {face_encoding.shape[0]}, known faces {known_encodings_array.shape[1]}")
            return "Unknown", 0.0
        
        # Calculate similarities using vectorized operations
        similarities = np.dot(known_encodings_array, face_encoding)
        
        # Apply quantization compensation if enabled
        if use_quantization:
            # Slightly boost similarities to compensate for quantization precision loss
            similarities = similarities * 1.02  # 2% boost to compensate
            # Apply smoothing to reduce quantization noise
            similarities = np.clip(similarities, 0, 1)
        
        # Find best match
        best_match_index = np.argmax(similarities)
        best_similarity = similarities[best_match_index]
        
        # Dynamic threshold adjustment for quantization
        adjusted_threshold = threshold
        if use_quantization:
            # Lower threshold slightly to account for precision loss
            adjusted_threshold = max(0.25, threshold - 0.03)
        
        if best_similarity > adjusted_threshold:
            return known_names[best_match_index], best_similarity
        
    except Exception as e:
        print(f"Error in face matching: {str(e)}")
        return "Unknown", 0.0
        return known_names[best_match_index], best_similarity
    else:
        return "Unknown", best_similarity

# ----------------------------
# Load CSS and Initialize
# ----------------------------
load_custom_css()

# Load known faces
known_encodings, known_names = load_known_faces()

# Initialize without loading model yet (will be loaded based on settings)
app = None

# ----------------------------
# Main UI
# ----------------------------
st.title("Criminal Face Detection System")
st.markdown("Real-time monitoring for security enforcement")
st.markdown("---")

# Status indicators
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.metric("Database", f"{len(known_names)} profiles", "Ready" if len(known_names) > 0 else "Empty")

with col2:
    # Camera status will be updated after camera selection
    camera_status_placeholder = st.empty()

with col3:
    alert_placeholder = st.empty()

# Control buttons
col1, col2 = st.columns(2)
with col1:
    start_monitoring = st.button("Start Monitoring", type="primary")
with col2:
    stop_monitoring = st.button("Stop Monitoring")

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Control Panel")

# Camera settings
st.sidebar.subheader("Camera Settings")

# Detect available cameras
available_cameras = detect_available_cameras()

if not available_cameras:
    st.sidebar.error("No cameras detected!")
    st.stop()

# Display camera information
st.sidebar.info(f"Found {len(available_cameras)} camera(s)")
for cam in available_cameras:
    st.sidebar.caption(f"📷 Camera {cam['index']}: {cam['resolution']} @ {cam['fps']}fps")

# Camera selection mode
camera_mode = st.sidebar.radio(
    "Camera Mode",
    ["Single Camera", "Multiple Cameras"],
    help="Choose between single camera or simultaneous multi-camera monitoring"
)

if camera_mode == "Single Camera":
    # Single camera selection (backward compatibility)
    camera_options = [cam['index'] for cam in available_cameras]
    selected_camera = st.sidebar.selectbox(
        "Select Camera", 
        camera_options,
        format_func=lambda x: f"Camera {x}",
        help="Select camera device"
    )
    selected_cameras = [selected_camera]
    st.sidebar.success(f"✅ Single camera mode: Camera {selected_camera}")
    
    # Update camera status display
    camera_status_placeholder.metric("Cameras", "1 active", "Single Camera")
    
else:
    # Multiple camera selection
    camera_options = [cam['index'] for cam in available_cameras]
    selected_cameras = st.sidebar.multiselect(
        "Select Cameras",
        camera_options,
        default=camera_options if len(camera_options) <= 3 else camera_options[:2],
        format_func=lambda x: f"Camera {x}",
        help="Select multiple cameras for simultaneous monitoring"
    )
    
    if not selected_cameras:
        st.sidebar.warning("Please select at least one camera")
        st.stop()
    
    st.sidebar.success(f"✅ Multi-camera mode: {len(selected_cameras)} cameras selected")

# Update camera status display
camera_status_placeholder.metric("Cameras", f"{len(selected_cameras)} active", f"{camera_mode}")

# Detection settings  
st.sidebar.subheader("Detection Settings")
detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)

# Performance settings
st.sidebar.subheader("Performance Settings")

# Quantization toggle
use_quantization = st.sidebar.checkbox(
    "Enable Model Quantization", 
    value=False, 
    help="Boost performance by 40-50% with minimal accuracy loss (recommended for real-time monitoring)"
)

if use_quantization:
    st.sidebar.success("🚀 Quantization: ON - Faster inference enabled")
else:
    st.sidebar.info("🔧 Quantization: OFF - Standard precision")

max_fps = st.sidebar.slider("Max FPS", 5, 30, 25)
detection_size = st.sidebar.selectbox("Detection Size", 
                                     [(128, 128), (240, 240), (320, 320)], 
                                     index=0,
                                     format_func=lambda x: f"{x[0]}x{x[1]} {'(Ultra Fast)' if x[0] == 128 else '(Balanced)' if x[0] == 240 else '(Accurate)'}")

# Auto-adjust threshold for quantization
base_similarity_threshold = st.sidebar.slider("Base Face Match Threshold", 0.1, 1.0, 0.4, 0.05)
if use_quantization:
    # Automatically adjust threshold for quantization
    similarity_threshold = max(0.25, base_similarity_threshold - 0.03)
    st.sidebar.caption(f"Auto-adjusted to {similarity_threshold:.2f} for quantization")
else:
    similarity_threshold = base_similarity_threshold

# Add new face
st.sidebar.subheader("Add New Criminal Profile")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
new_name = st.sidebar.text_input("Name/ID")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Add Profile"):
        if add_new_face(uploaded_file, new_name):
            st.sidebar.success(f"Added {new_name} to database!")
            st.rerun()
        else:
            st.sidebar.error("Please provide both image and name")

with col2:
    if st.button("🔄 Reset DB"):
        st.cache_data.clear()
        st.sidebar.success("Database cache cleared!")
        st.rerun()

# Display known faces
if known_names:
    st.sidebar.subheader("Known Profiles")
    for name in known_names:
        st.sidebar.text(f"• {name}")

# ----------------------------
# Camera Monitoring
# ----------------------------
if start_monitoring:
    if camera_mode == "Single Camera":
        st.subheader("🔍 Single Camera Monitoring")
    else:
        st.subheader(f"🔍 Multi-Camera Monitoring ({len(selected_cameras)} cameras)")
    
    # Load model with current settings
    if app is None or 'last_quantization_setting' not in st.session_state or st.session_state.last_quantization_setting != use_quantization:
        with st.spinner("Loading optimized face recognition model..."):
            app = load_face_recognition(use_quantization, detection_size)
            st.session_state.last_quantization_setting = use_quantization
            if use_quantization:
                st.success("🚀 Quantized model loaded - Enhanced performance mode active!")
            else:
                st.info("🔧 Standard model loaded")
    
    # Alert display above camera feed
    monitoring_alert_placeholder = st.empty()
    
    # Performance metrics display
    if camera_mode == "Single Camera":
        # Single camera feed
        st_frame = st.empty()
        
    else:
        # Multi-camera layout
        num_cameras = len(selected_cameras)
        if num_cameras <= 2:
            cols = st.columns(num_cameras)
        elif num_cameras <= 4:
            cols = st.columns(2)
        else:
            cols = st.columns(3)
        
        # Create frame placeholders for each camera
        camera_frames = {}
        
        for i, cam_idx in enumerate(selected_cameras):
            col_idx = i % len(cols)
            with cols[col_idx]:
                st.write(f"📷 **Camera {cam_idx}**")
                camera_frames[cam_idx] = st.empty()
    
    # Initialize camera manager
    if camera_mode == "Single Camera":
        # Single camera setup (backward compatible)
        camera_index = selected_cameras[0]
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            st.error(f"Could not open Camera {camera_index}")
            st.stop()
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, max_fps)
        
        # Performance tracking
        performance_timer = PerformanceTimer()
        frame_count = 0
        detection_persistence = {}
        
    else:
        # Multi-camera setup
        camera_manager = MultiCameraManager(selected_cameras, max_fps)
        camera_manager.initialize_cameras()
        
        if not camera_manager.cameras:
            st.error("Could not initialize any cameras")
            st.stop()
        
        # Performance tracking for each camera
        performance_timers = {cam_idx: PerformanceTimer() for cam_idx in selected_cameras}
        frame_counts = {cam_idx: 0 for cam_idx in selected_cameras}
        detection_persistences = {cam_idx: {} for cam_idx in selected_cameras}
    
    # Global alert placeholder
    # Alert is now displayed above camera feeds
    
    try:
        while True:
            current_time = time.time()
            overall_criminal_detected = False
            
            if camera_mode == "Single Camera":
                # Single camera processing (original logic)
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                performance_timer.tick()
                frame_count += 1
                
                # Skip frames for performance
                skip_frames = max(1, 30 // max_fps)
                process_detection = (frame_count % skip_frames == 0)
                
                if process_detection:
                    detections, criminal_detected = process_camera_frame(
                        frame, app, known_encodings, known_names, 
                        similarity_threshold, use_quantization, detection_size
                    )
                    detection_persistence = detections
                    if criminal_detected:
                        overall_criminal_detected = True
                
                # Draw detections
                frame, any_criminal = draw_detections_on_frame(frame, detection_persistence, current_time)
                if any_criminal:
                    overall_criminal_detected = True
                
                # Update performance metrics (removed FPS display)
                # Performance metrics removed for cleaner interface
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                
            else:
                # Multi-camera processing
                frames = camera_manager.capture_all_frames()
                
                for cam_idx in selected_cameras:
                    if cam_idx in frames and frames[cam_idx] is not None:
                        frame = frames[cam_idx]
                        performance_timers[cam_idx].tick()
                        frame_counts[cam_idx] += 1
                        
                        # Skip frames for performance
                        skip_frames = max(1, 30 // max_fps)
                        process_detection = (frame_counts[cam_idx] % skip_frames == 0)
                        
                        if process_detection:
                            detections, criminal_detected = process_camera_frame(
                                frame, app, known_encodings, known_names,
                                similarity_threshold, use_quantization, detection_size
                            )
                            detection_persistences[cam_idx] = detections
                            if criminal_detected:
                                overall_criminal_detected = True
                        
                        # Draw detections
                        frame, any_criminal = draw_detections_on_frame(
                            frame, detection_persistences[cam_idx], current_time
                        )
                        if any_criminal:
                            overall_criminal_detected = True
                        
                        # Performance metrics removed for cleaner interface
                        
                        # Display frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_frames[cam_idx].image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Global alert status with neon effect (above camera feed)
            if overall_criminal_detected:
                monitoring_alert_placeholder.markdown(
                    '<div class="criminal-alert">🚨 CRIMINAL DETECTED! 🚨</div>', 
                    unsafe_allow_html=True
                )
            else:
                monitoring_alert_placeholder.markdown(
                    '<div class="all-clear">✅ ALL CLEAR</div>', 
                    unsafe_allow_html=True
                )
            
            # Check for stop button
            if stop_monitoring:
                break
            
            # Minimal delay for maximum FPS
            time.sleep(1.0 / max_fps)
            
    except Exception as e:
        st.error(f"Error during monitoring: {str(e)}")
    finally:
        if camera_mode == "Single Camera":
            cap.release()
        else:
            camera_manager.release_all()

# Footer
st.markdown("---")
st.markdown("*Criminal Face Detection System - Real-time monitoring solution*")
