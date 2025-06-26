import os
import uuid

import cv2
from insightface.app import FaceAnalysis

from util.config import GLOBAL_CONFIG
from util.logger import logger
from util.pineconeutil import initialize_pinecone


def load_pinecone_embeddings_from_local_dir(face_images_dir):
    """
    Load face embeddings from local storage and upload to Pinecone
    """
    try:
        # Initialize Pinecone

        logger.info("Pinecone initialized successfully")
        pc = initialize_pinecone()
        # Get index name from config

        index_name = GLOBAL_CONFIG.pinecone.face_embeddings_index_nm

        index = pc.Index(index_name)

        # Initialize face analysis model
        logger.debug("Initializing face analysis model...")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0)
        logger.debug("Face analysis model initialized")

        # Load images and extract embeddings

        if not os.path.exists(face_images_dir):
            raise FileNotFoundError(f"Directory {face_images_dir} not found")

        upserts = []
        processed_images = 0
        failed_images = 0

        logger.info(f"Processing images from {face_images_dir}...")
        for person_name in os.listdir(face_images_dir):
            person_dir = os.path.join(face_images_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            logger.info(f"Processing person: {person_name}")
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        logger.error(f"Failed to load image: {img_path}")
                        failed_images += 1
                        continue

                    faces = app.get(img)
                    if not faces:
                        logger.debug(f"No faces detected in: {img_path}")
                        failed_images += 1
                        continue

                    emb = faces[0].embedding.tolist()
                    if not hasattr(faces[0], "embedding") or len(faces[0].embedding) != 512:
                        logger.error(f"Invalid embedding from: {img_path}")
                        failed_images += 1
                        continue

                    uid = str(uuid.uuid4())
                    meta = {
                        "name": person_name,
                        "img_path": img_path
                    }
                    upserts.append((uid, emb, meta))
                    processed_images += 1

                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {str(e)}")
                    failed_images += 1
                    continue

        # Upsert in batches
        if upserts:
            batch_size = 100
            for i in range(0, len(upserts), batch_size):
                batch = upserts[i:i + batch_size]
                try:
                    index.upsert(vectors=batch)
                    logger.info(f"Uploaded batch {i // batch_size + 1}/{(len(upserts) - 1) // batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error uploading batch to Pinecone: {str(e)}")

        logger.info("\nProcess completed:")
        logger.info(f"Successfully processed: {processed_images} images")
        logger.info(f"Failed to process: {failed_images} images")
        logger.info(f"Uploaded {len(upserts)} embeddings to Pinecone index '{index_name}'")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    load_pinecone_embeddings_from_local_dir("faces_db")
