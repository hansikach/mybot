import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import uuid
from create_face_embeddings import load_pinecone_embeddings_from_local_dir

from com.hansiz.bot.util.logger import logger

# Configure
OUTPUT_DIR = "faces_db"
POSE_THRESHOLD_DEGREES = 15  # accept if new pose differs this much
MAX_IMAGES = 20

def angular_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def enroll_from_video(user_name):
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    user_dir = os.path.join(OUTPUT_DIR, user_name)
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    collected_poses = []
    saved = 0

    logger.info("Look at the camera from different angles. Press 'q' to quit early.")

    while cap.isOpened() and saved < MAX_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        if not faces:
            continue

        face = faces[0]
        pose = face.pose  # (yaw, pitch, roll)
        is_diverse = all(angular_distance(pose, prev) > POSE_THRESHOLD_DEGREES for prev in collected_poses)

        if is_diverse:
            uid = str(uuid.uuid4())
            path = os.path.join(user_dir, f"{uid}.jpg")
            cv2.imwrite(path, frame)
            collected_poses.append(pose)
            saved += 1
            logger.info(f"Saved frame {saved} with pose {tuple(round(p) for p in pose)}")

        cv2.imshow("Enroll View", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Enrollment finished for {user_name}. {saved} diverse frames saved to {user_dir}")

if __name__ == "__main__":
    username = "prasanth2"

    enroll_from_video(username)

    load_pinecone_embeddings_from_local_dir(OUTPUT_DIR)