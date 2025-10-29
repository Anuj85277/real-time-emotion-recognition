# 🧠 Real-Time Emotion Detection — Short Summary

I built a **Real-Time Emotion Detection System** using **Python, TensorFlow, and OpenCV**.

The system captures faces through a webcam, detects them using the **MTCNN** face detector, and predicts emotions using a **CNN model** trained on the **FER-2013 dataset**.

It recognizes seven emotions:
😄 Happy | 😠 Angry | 😢 Sad | 😐 Neutral | 😲 Surprise | 😨 Fear | 🤢 Disgust

### 🔧 Tech Stack:
- **Python**
- **TensorFlow/Keras** — Deep Learning Model
- **OpenCV** — Real-time webcam processing
- **MTCNN** — Face detection
- **NumPy, Matplotlib** — Preprocessing & visualization

### ⚙️ Working:
1. Capture frame from webcam.  
2. Detect face using MTCNN.  
3. Preprocess face (grayscale, resize 48×48, normalize).  
4. Predict emotion using CNN model (`emotion_model.h5`).  
5. Display emotion label live on webcam feed.

### 💡 Key Learnings:
- Built and trained a CNN from scratch.
- Integrated OpenCV with deep learning.
- Learned real-time video processing and model deployment.

### 🧩 Applications:
- Student engagement detection
- Customer emotion tracking
- Adaptive gaming and marketing systems
