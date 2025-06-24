import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

from com.hansiz.bot.util.config import GLOBAL_CONFIG
from com.hansiz.bot.util.logger import logger
from com.hansiz.bot.util.pineconeutil import initialize_pinecone


def recognize_faces():
    """
    Recognize faces using embeddings stored in Pinecone.
    """
    # Initialize Pinecone
    pc = initialize_pinecone()
    # Get index name from config
    index_name = GLOBAL_CONFIG.pinecone.face_embeddings_index_nm
    index = pc.Index(index_name)

    # Add this after index initialization to debug
    stats = index.describe_index_stats()
    logger.info(f"Index stats: {stats}")

    # --- Load known embeddings from Pinecone ---
    logger.info("Fetching existing face embeddings from Pinecone...")
    names = []
    embeddings = []

    # If you stored metadata 'name', use it for labeling
    data = index.query(vector=[0.0] * 512, top_k=1000, include_metadata=True, include_values=True)  # dummy vector to fetch all

    for match in data.get("matches", []):
        emb = match["values"]
        #if not emb or len(emb) != 512:
        #    continue  # skip invalid vectors

        meta = match.get("metadata", {})
        name = meta.get("name", "Unknown")
        embeddings.append(np.array(emb))
        names.append(name)

    if not embeddings:
        raise ValueError("No face embeddings found in Pinecone.")

    logger.info(f"Loaded {len(embeddings)} identities.")

    # --- Init face analysis ---
    app = FaceAnalysis(name='buffalo_l', 
                  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    # --- Webcam Recognition ---
    cap = cv2.VideoCapture(0)
    frame_count = 0
    process_every_n = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % process_every_n != 0:
            continue

        faces = app.get(frame)

        #embeddings = [e for e in embeddings if e.shape == (512,)]
        #if not embeddings:
        #   raise ValueError("All fetched embeddings are empty or invalid.")

        for face in faces:
            emb = face.embedding.reshape(1, -1)
            sims = cosine_similarity(emb, np.vstack(embeddings))
            best_idx = int(np.argmax(sims))
            score = sims[0][best_idx]
            label = names[best_idx] if score > GLOBAL_CONFIG.face_match_threshold else "Unknown"

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