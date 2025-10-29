import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Load the trained model
model = load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam and MTCNN detector
cap = cv2.VideoCapture(0)  # 0 = default camera
detector = MTCNN()

print("Starting real-time emotion detection... Press 'q' to quit.")

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert frame to RGB (for MTCNN)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    # Loop through detected faces
    for face in faces:
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)
        face_img = rgb_frame[y:y + height, x:x + width]

        try:
            # Preprocess face for model
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_resized = face_resized / 255.0
            face_resized = np.reshape(face_resized, (1, 48, 48, 1))

            # Predict emotion
            prediction = model.predict(face_resized, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]

            # Draw rectangle & label around face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        except Exception as e:
            print("Error processing face:", e)

    # Display the frame
    cv2.imshow("Real-time Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Real-time emotion detection stopped.")
