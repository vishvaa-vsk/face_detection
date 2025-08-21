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
    </style>
    """, unsafe_allow_html=True)

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

# ----------------------------
# Cache face recognition app and models
# ----------------------------
@st.cache_resource
def load_face_recognition():
    """Load and cache the face recognition model"""
    app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(128, 128))  # Ultra performance mode
    return app

@st.cache_data
def load_known_faces():
    """Load known faces from the face_db directory"""
    known_encodings = []
    known_names = []
    face_db_path = "face_db"
    
    if not os.path.exists(face_db_path):
        os.makedirs(face_db_path)
        return known_encodings, known_names
    
    app = load_face_recognition()
    
    for filename in os.listdir(face_db_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(face_db_path, filename)
            try:
                # Load image using CV2
                img = cv2.imread(image_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Get face embeddings
                    faces = app.get(img_rgb)
                    if faces:
                        # Use the first detected face
                        face_encoding = faces[0].embedding
                        known_encodings.append(face_encoding)
                        # Use filename without extension as name
                        name = os.path.splitext(filename)[0]
                        known_names.append(name)
            except Exception as e:
                st.warning(f"Could not process {filename}: {str(e)}")
    
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

def find_face_matches(face_encoding, known_encodings, known_names, threshold=0.4):
    """Find matches for a face encoding"""
    if len(known_encodings) == 0:
        return "Unknown", 1.0
    
    # Calculate similarities using vectorized operations
    known_encodings_array = np.array(known_encodings)
    similarities = np.dot(known_encodings_array, face_encoding)
    
    # Find best match
    best_match_index = np.argmax(similarities)
    best_similarity = similarities[best_match_index]
    
    if best_similarity > threshold:
        return known_names[best_match_index], best_similarity
    else:
        return "Unknown", best_similarity

# ----------------------------
# Load CSS and Initialize
# ----------------------------
load_custom_css()

# Load known faces
known_encodings, known_names = load_known_faces()
app = load_face_recognition()

# ----------------------------
# Main UI
# ----------------------------
st.title("Criminal Face Detection System")
st.markdown("Real-time monitoring for security enforcement")
st.markdown("---")

# Status indicators
col1, col2 = st.columns([1, 3])

with col1:
    st.metric("Database", f"{len(known_names)} profiles", "Ready" if len(known_names) > 0 else "Empty")

with col2:
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
camera_index = st.sidebar.selectbox("Camera", [0, 1, 2], help="Select camera device")

# Detection settings  
st.sidebar.subheader("Detection Settings")
detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)
similarity_threshold = st.sidebar.slider("Face Match Threshold", 0.1, 1.0, 0.4, 0.05)

# Performance settings
st.sidebar.subheader("Performance Settings")
max_fps = st.sidebar.slider("Max FPS", 5, 30, 25)
detection_size = st.sidebar.selectbox("Detection Size", 
                                     [(128, 128), (240, 240), (320, 320)], 
                                     index=0,
                                     format_func=lambda x: f"{x[0]}x{x[1]} {'(Ultra Fast)' if x[0] == 128 else '(Balanced)' if x[0] == 240 else '(Accurate)'}")

# Add new face
st.sidebar.subheader("Add New Criminal Profile")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
new_name = st.sidebar.text_input("Name/ID")
if st.sidebar.button("Add Profile"):
    if add_new_face(uploaded_file, new_name):
        st.sidebar.success(f"Added {new_name} to database!")
        st.rerun()
    else:
        st.sidebar.error("Please provide both image and name")

# Display known faces
if known_names:
    st.sidebar.subheader("Known Profiles")
    for name in known_names:
        st.sidebar.text(f"• {name}")

# ----------------------------
# Camera Monitoring
# ----------------------------
if start_monitoring:
    st.subheader("Live Camera Feed")
    
    # Create placeholder for video
    st_frame = st.empty()
    
    # Initialize camera with optimized settings
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("Could not open camera")
        st.stop()
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, max_fps)
    
    # Performance tracking
    performance_timer = PerformanceTimer()
    frame_count = 0
    last_detection_time = 0
    detection_persistence = {}  # Store detection results to reduce blinking
    
    # Update detection model size
    app.prepare(ctx_id=0, det_size=detection_size)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            performance_timer.tick()
            frame_count += 1
            
            # Skip frames for performance (process every nth frame based on FPS)
            skip_frames = max(1, 30 // max_fps)
            process_detection = (frame_count % skip_frames == 0)
            
            current_time = time.time()
            criminal_detected = False
            
            if process_detection:
                # Resize frame for faster processing
                resize_factor = 0.5 if detection_size[0] <= 240 else 0.7
                small_frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Face detection
                faces = app.get(rgb_frame)
                
                # Update detection persistence
                current_detections = {}
                
                for face in faces:
                    # Scale coordinates back to original frame
                    bbox = (face.bbox / resize_factor).astype(int)
                    face_encoding = face.embedding
                    
                    # Find match
                    name, similarity = find_face_matches(face_encoding, known_encodings, known_names, similarity_threshold)
                    
                    # Store detection for persistence
                    detection_key = f"{bbox[0]}_{bbox[1]}"
                    current_detections[detection_key] = {
                        'bbox': bbox,
                        'name': name,
                        'similarity': similarity,
                        'timestamp': current_time
                    }
                    
                    if name != "Unknown":
                        criminal_detected = True
                
                # Update persistence dict
                detection_persistence = current_detections
                last_detection_time = current_time
            
            # Draw persistent detections (reduces blinking)
            for detection in detection_persistence.values():
                if current_time - detection['timestamp'] < 1.0:  # Show for 1 second
                    bbox = detection['bbox']
                    name = detection['name']
                    similarity = detection['similarity']
                    
                    if name != "Unknown":
                        criminal_detected = True
                    
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
            
            # Update status less frequently
            if frame_count % 10 == 0:
                # Alert status only
                if criminal_detected:
                    alert_placeholder.error("🚨 Criminal Detected!")
                else:
                    alert_placeholder.success("✅ All Clear")
            
            # Convert and display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Check for stop button or if monitoring should stop
            if stop_monitoring:
                break
            
            # Minimal delay for maximum FPS
            time.sleep(1.0 / max_fps)
            
    except Exception as e:
        st.error(f"Error during monitoring: {str(e)}")
    finally:
        cap.release()

# Footer
st.markdown("---")
st.markdown("*Criminal Face Detection System - Real-time monitoring solution*")
