                               Real-Time Emotion Detection using CNN and OpenCV


🧠 1. Project Overview

The goal of this project is to detect human emotions in real-time from facial expressions captured through a webcam or an image.

The model identifies emotions such as:
😠 Angry, 🤢 Disgust, 😨 Fear, 😄 Happy, 😢 Sad, 😲 Surprise, 😐 Neutral

It combines Deep Learning (CNN) for emotion classification and Computer Vision (OpenCV + MTCNN) for face detection.

🔍 2. Problem Statement

Humans can easily recognize emotions through facial expressions — but can a computer do the same?
This project aims to answer that by building a model that can analyze faces and predict emotions automatically.

Such systems can be used in:

Emotion-based recommendation systems (e.g., ads, content)

Online education (detect student engagement)

Customer experience monitoring

Human-computer interaction

3. System Architecture

Here’s the high-level flow:

Webcam / Image Input
        ↓
Face Detection (MTCNN)
        ↓
Preprocessing (Grayscale, Resize 48×48)
        ↓
Emotion Classification (CNN Model)
        ↓
Predicted Emotion Display (OpenCV Frame)

🧱 4. Tools and Technologies Used
Component	Technology
Programming Language	Python
Deep Learning Framework	TensorFlow / Keras
Face Detection	MTCNN (Multi-Task Cascaded Convolutional Networks)
Computer Vision	OpenCV
Visualization	Matplotlib
Dataset (for training)	FER-2013 (Facial Expression Recognition dataset)
🧮 5. Working Steps
Step 1 — Face Detection

We use MTCNN to detect faces in an image or video frame.

It returns bounding box coordinates of faces.

This ensures that only the face region is passed for emotion detection.

Step 2 — Preprocessing

Extract the detected face region.

Convert it to grayscale (as the model is trained on grayscale images).

Resize the image to 48×48 pixels.

Normalize pixel values to range [0,1].

Step 3 — Model Prediction

The CNN model (emotion_model.h5) takes the processed face as input.

It predicts probabilities for each of the 7 emotions.

The emotion with the highest probability is chosen as the final output.

Step 4 — Display Result

OpenCV draws a bounding box around the face.

The predicted emotion is displayed above the face in real-time.

🧠 6. Deep Learning Model (CNN)

A simple Convolutional Neural Network was trained on the FER-2013 dataset.

Model Summary:

Input layer: (48, 48, 1)

Convolution layers: 3 layers with ReLU activation

Pooling layers: MaxPooling2D after each convolution

Fully Connected layer: Dense + Dropout to prevent overfitting

Output layer: Softmax activation (7 classes)

Loss Function & Optimizer:

Loss: Categorical Crossentropy

Optimizer: Adam

Accuracy Metric: Validation accuracy

🧪 7. Real-Time Execution Flow

The webcam captures continuous frames.

MTCNN detects all faces in each frame.

For each detected face:

The face is cropped, preprocessed, and passed to the CNN model.

Model predicts the emotion.

The emotion label is drawn on the frame.

The processed video stream is displayed live with emotion overlays.

📊 8. Results

When you run the real-time script:

The webcam feed opens.

The model detects your face and continuously predicts your emotion.

Example output:

Detected Emotion: Happy 😀
Detected Emotion: Neutral 😐
Detected Emotion: Surprise 😲


It works for single or multiple faces in the same frame.

🧰 9. Project Files
File	Description
emotion_model.h5	Trained CNN model for emotion detection
test_model.py	Detects emotion from a static image
real_time_emotion.py	Real-time emotion detection using webcam
requirements.txt	List of dependencies
sample.jpg	Sample image for testing

10. Installation & Setup
Step 1: Clone or create the folder
git clone <your_repo_link>
cd Face\ detection

Step 2: Create virtual environment
python -m venv env
env\Scripts\activate

Step 3: Install dependencies
pip install -r requirements.txt

Step 4: Run the model on image
python test_model.py

Step 5: Run the model in real time
python real_time_emotion.py

🌟 11. Applications
Area	Use Case
Education	Detect student engagement or boredom
Marketing	Real-time audience emotion analysis
Healthcare	Detect depression or stress from facial cues
Gaming	Adaptive difficulty based on player emotions
Customer Service	Improve chatbot empathy and feedback systems
🧠 12. Key Learning Outcomes

Hands-on experience with Deep Learning and CNNs

Working knowledge of MTCNN for face detection

Real-time video stream handling using OpenCV

Model deployment concepts

Integration of multiple AI components in one system