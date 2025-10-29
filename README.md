# Real-Time Emotion Recognition

A Python-based project that detects faces in images and predicts emotions in real-time using deep learning. This project leverages **MTCNN** for face detection and a **CNN model** trained on FER2013 dataset for emotion recognition.

---

## 🔹 Features

- Detects faces in images using **MTCNN**
- Predicts one of seven emotions:  
  `Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`
- Real-time or image-based emotion detection
- Outputs results visually with **Matplotlib**
- Lightweight and easy to extend

---

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **TensorFlow / Keras** – deep learning for emotion recognition
- **MTCNN** – for accurate face detection
- **OpenCV** – image processing
- **NumPy** – numerical computations
- **Matplotlib** – display detected faces and predictions

---

## ⚡ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Anuj85277/real-time-emotion-recognition.git
cd real-time-emotion-recognition
Create and activate virtual environment

bash
Copy code
python -m venv env
# Windows
.\env\Scripts\activate
# macOS/Linux
source env/bin/activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the emotion detection script

bash
Copy code
python real_time_emotion.py
Optional: Test on a single image

bash
Copy code
python test_model.py
📂 Project Structure
bash
Copy code
Face detection/
├─ env/                     # Virtual environment
├─ fer2013/                 # Dataset folder
├─ emotion_model.h5         # Trained emotion recognition model
├─ real_time_emotion.py     # Real-time emotion detection script
├─ test_model.py            # Script to test single images
├─ train_emotion_model.py   # Model training script
├─ requirements.txt         # Python dependencies
├─ README.md                # Project documentation
📊 Supported Emotions
Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

⚡ Usage Example
After running the script on an image:


The program will display the detected face with the predicted emotion on top.

🔐 Notes
Ensure the image path is correct when testing on individual images.

For large datasets or images, using a GPU is recommended for faster processing.

.gitignore is configured to ignore large files and environment folders.

📌 License
This project is open-source under the MIT License.
