import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

from util.config import GLOBAL_CONFIG
from util.logger import logger
from util.pineconeutil import initialize_pinecone

import warnings


warnings.filterwarnings("ignore", message="`rcond` parameter will change")
warnings.filterwarnings("ignore", message=".*ONNX Runtime supports Windows 10 and above.*",
    category=UserWarning
)


def recognize_faces():
    """
    Recognize faces using embeddings stored in Pinecone.
    """
    # Initialize Pinecone
    global label, score, face
    pc = initialize_pinecone()
    # Get index name from config
    index_name = GLOBAL_CONFIG.pinecone.face_embeddings_index_nm
    index = pc.Index(index_name)

    # --- Load known embeddings from Pinecone ---
    logger.info("Fetching existing face embeddings from Pinecone...")
    names = []
    embeddings = []

    # If you stored metadata 'name', use it for labeling
    data = index.query(vector=[0.0] * 512, top_k=1000, include_metadata=True,
                       include_values=True)  # dummy vector to fetch all

    for match in data.get("matches", []):
        emb = match["values"]
        meta = match.get("metadata", {})
        name = meta.get("name", "Unknown")
        embeddings.append(np.array(emb))
        names.append(name)

    if not embeddings:
        raise ValueError("No face embeddings found in Pinecone.")

    logger.info(f"Loaded {len(embeddings)} identities.")

    # --- Init face analysis ---
    app = FaceAnalysis(name='buffalo_l',
                       providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    # --- Webcam Recognition ---
    cap = cv2.VideoCapture(0)
    frame_count = 0
    process_every_n = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % process_every_n == 0:

            faces = app.get(frame)

            if not faces:
                logger.info("No faces detected.")
                continue

            for face in faces:
                emb = face.embedding.reshape(1, -1)
                sims = cosine_similarity(emb, np.vstack(embeddings))
                best_idx = int(np.argmax(sims))
                score = sims[0][best_idx]
                label = names[best_idx] if score > GLOBAL_CONFIG.face_match_threshold else "Unknown"
                if label != "Unknown":
                    logger.info(f"Best match: {label} ({score:.2f})")

            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Pinecone Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_faces()
