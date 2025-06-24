import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

from com.hansiz.bot.util.config import GLOBAL_CONFIG
from com.hansiz.bot.util.logger import logger
from com.hansiz.bot.util.pineconeutil import initialize_pinecone

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
    pc = initialize_pinecone()
    # Get index name from config
    index_name = GLOBAL_CONFIG.pinecone.face_embeddings_index_nm
    index = pc.Index(index_name)


    # --- Init face analysis ---
    app = FaceAnalysis(name='buffalo_l',
                       providers=['CPUExecutionProvider'])
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
        if frame_count % process_every_n == 0:

            faces = app.get(frame)

            for face in faces:
                emb = face.embedding.reshape(1, -1)
                response = index.query(
                    vector=emb.flatten().tolist(),
                    top_k=10,
                    include_metadata=True,
                    namespace=""  # or your specific namespace if used
                )

                matches = response.get("matches", [])
                if matches and matches[0]["score"] > GLOBAL_CONFIG.face_match_threshold:
                    label = matches[0]["metadata"].get("name", "Unknown")
                    score = matches[0]["score"]
                else:
                    label = "Unknown"
                    score = 0.0
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
