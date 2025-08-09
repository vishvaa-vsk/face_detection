import streamlit as st
import cv2
import os
import numpy as np
import tempfile
import insightface
from PIL import Image

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

model = load_model()

# ----------------------------
# Utility Functions
# ----------------------------
def get_face_embedding(img_path):
    img = cv2.imread(img_path)
    faces = model.get(img)
    if len(faces) > 0:
        return faces[0].normed_embedding
    return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------
# Sidebar Config
# ----------------------------
st.sidebar.title("Face Matching Settings")
db_path = st.sidebar.text_input("Database Folder", value="face_db")
threshold = st.sidebar.slider("Match Threshold", 0.1, 0.8, 0.35, 0.01)

# Upload new face option
uploaded_face = st.sidebar.file_uploader("Add New Face to DB", type=["jpg", "jpeg", "png"])
if uploaded_face is not None:
    save_path = os.path.join(db_path, uploaded_face.name)
    os.makedirs(db_path, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_face.getbuffer())
    st.sidebar.success(f"Added {uploaded_face.name} to DB!")

# ----------------------------
# Load DB embeddings
# ----------------------------
@st.cache_data
def load_db_embeddings(db_folder):
    known_embeddings = []
    known_names = []
    if not os.path.exists(db_folder):
        return np.array([]), []
    for file in os.listdir(db_folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            emb = get_face_embedding(os.path.join(db_folder, file))
            if emb is not None:
                known_embeddings.append(emb)
                known_names.append(os.path.splitext(file)[0])
    return np.array(known_embeddings), known_names

known_embeddings, known_names = load_db_embeddings(db_path)
st.sidebar.write(f"Loaded {len(known_names)} faces from DB.")

# ----------------------------
# Main UI
# ----------------------------
st.title("📸 Real-Time Face Matching")
st.write("Red box = Match | Green box = Unknown")

start_camera = st.button("Start Camera")

if start_camera:
    cap = cv2.VideoCapture(0)
    st_frame = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = model.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.normed_embedding

            if len(known_embeddings) > 0:
                similarities = [cosine_similarity(embedding, db_emb) for db_emb in known_embeddings]
                best_match_idx = int(np.argmax(similarities))
                best_score = similarities[best_match_idx]

                if best_score > threshold:
                    name = known_names[best_match_idx]
                    color = (0, 0, 255)  # Red
                else:
                    name = "Unknown"
                    color = (0, 255, 0)  # Green
            else:
                name = "Unknown"
                color = (0, 255, 0)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"{name}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB")

    cap.release()
