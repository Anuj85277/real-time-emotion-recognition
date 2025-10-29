import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Path to your image
image_path = r"D:\anuj\Face detection\WhatsApp Image 2025-10-17 at 11.26.32 PM.jpeg"

# Load image
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Image not found at {image_path}. Check the file path!")

# Convert to RGB (MTCNN expects RGB)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Initialize MTCNN face detector
detector = MTCNN()
faces = detector.detect_faces(rgb_img)

if len(faces) == 0:
    raise ValueError("No face detected in the image!")

# Take the first detected face
x, y, width, height = faces[0]['box']
x, y = max(0, x), max(0, y)
face_img = rgb_img[y:y+height, x:x+width]

# Preprocess face for model
face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
face_resized = cv2.resize(face_gray, (48, 48))
face_resized = face_resized / 255.0
face_resized = np.reshape(face_resized, (1, 48, 48, 1))

# Predict emotion
prediction = model.predict(face_resized)
emotion = emotion_labels[np.argmax(prediction)]
print("Predicted emotion:", emotion)

# Display the face with predicted emotion
plt.imshow(face_img)
plt.title(f"Predicted: {emotion}")
plt.axis('off')
plt.show()
