import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels (7 classes in FER2013)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MTCNN face detector
detector = MTCNN()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)
        face_img = rgb_frame[y:y+height, x:x+width]

        # Preprocess face for model
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img / 255.0
        face_img = np.reshape(face_img, (1, 48, 48, 1))

        # Predict emotion
        predictions = model.predict(face_img)
        emotion = emotion_labels[np.argmax(predictions)]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
