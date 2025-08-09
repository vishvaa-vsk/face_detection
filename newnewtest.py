import cv2
import os
import numpy as np
import insightface

DB_PATH = "face_db"

# Load the InsightFace model (includes detection + recognition)
model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

# Function to get embedding from image
def get_face_embedding(img_path):
    img = cv2.imread(img_path)
    faces = model.get(img)
    if len(faces) > 0:
        return faces[0].normed_embedding
    return None

# Load DB embeddings
known_embeddings = []
known_names = []

for file in os.listdir(DB_PATH):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        emb = get_face_embedding(os.path.join(DB_PATH, file))
        if emb is not None:
            known_embeddings.append(emb)
            known_names.append(os.path.splitext(file)[0])

known_embeddings = np.array(known_embeddings)
print(f"Loaded {len(known_names)} known faces.")

# Function to compute similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Start live video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = model.get(frame)
    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.normed_embedding

        # Compare with DB
        similarities = [cosine_similarity(embedding, db_emb) for db_emb in known_embeddings]
        if similarities:
            best_match_idx = int(np.argmax(similarities))
            best_score = similarities[best_match_idx]

            if best_score > 0.35:  # Threshold (tune for your case)
                name = known_names[best_match_idx]
                color = (0, 0, 255)  # Red for match
            else:
                name = "Unknown"
                color = (0, 255, 0)  # Green for no match
        else:
            name = "Unknown"
            color = (0, 255, 0)

        # Draw box & name
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, f"{name}", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Live Face Matching", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
