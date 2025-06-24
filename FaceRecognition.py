import os

import cv2
from deepface import DeepFace
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Path to known faces (organized by folders named after each person)
face_db_path = "faces_db"

# Load the webcam
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # For 720p use 1280x720
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print("[INFO] Starting face recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces with YOLO
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            if confidence > 0.6:
                if int(box.cls[0]) == 0:  # class 0 = person in YOLO
                    # proceed with recognition

                    # Crop and align the face
                    face_crop = frame[y1:y2, x1:x2]

                    try:
                        # DeepFace returns best match from folder
                        obj = DeepFace.find(
                            img_path=face_crop,
                            db_path=face_db_path,
                            model_name="ArcFace",
                            enforce_detection=False,
                            detector_backend="skip",  # We're using YOLO already
                            silent=True,
                            threshold=0.3
                        )

                        if len(obj) > 0 and not obj[0].empty:
                            matched_name = os.path.basename(obj[0]["identity"][0].split(os.sep)[-2])
                        else:
                            matched_name = "Unknown"

                    except Exception as e:
                        print("Recognition error:", e)
                        matched_name = "Unknown"

                    # Draw results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, matched_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
