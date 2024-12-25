Real-Time Face Recognition Using CNN

Overview

This project implements a basic real-time face recognition system using a Convolutional Neural Network (CNN). It detects faces via a webcam and recognizes them based on a trained model.
The Model is trained on face image of five famous people.

Setup Instructions

Step 1: Clone the Repository
git clone https://github.com/DagoTshering/Real-Time-Face-Recognition-Using-CNN.git
cd realtime-face-recognition

Step 2: Create a Virtual Environment
Create and activate a virtual environment to manage dependencies.

For Windows: 
python -m venv venv
venv\Scripts\activate

For macOS/Linux:
python3 -m venv venv
source venv/bin/activate

Step 3: Install Required Libraries
Install the following libraries:
numpy
opencv-contrib-python
opencv-python
tensorflow
keras
dlib
h5py
pillow
scikit-learn

Step 4: Running the files
First run 01_faceDataset.py files
Second run 02_training.py
Third run  03_faceRecognition.py
Fourth run 04_facePrediction.py

Acknowledgments
Thanks to the open-source community for tools like TensorFlow and OpenCV, Kaggel for the datasets, making this project possible.

Enjoy exploring face recognition! 😊