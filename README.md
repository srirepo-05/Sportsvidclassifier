# Sports Classification and Pose Estimation

## Overview

This project involves classifying sports activities using YOLOv5 and performing pose estimation on video data. It includes steps for setting up YOLOv5, training a model, visualizing training metrics, predicting sports from video, and applying pose estimation.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the YOLOv5 Model](#training-the-yolov5-model)
  - [Plotting Accuracy and Loss Graphs](#plotting-accuracy-and-loss-graphs)
  - [Predicting Sports](#predicting-sports)
  - [Pose Estimation](#pose-estimation)
- [Code Structure](#code-structure)

## Features

- **Sports Classification:** Train a YOLOv5 model to classify different sports activities.
- **Pose Estimation:** Extract and visualize pose landmarks from sports videos.
- **Visualization:** Plot accuracy and loss metrics during training.

## Installation

1. **Clone the Repositories:**
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   ```

2. **Unzip the Dataset:**
   Ensure that your dataset is available in the appropriate directory. The code expects the dataset to be unzipped in the `Sports_Classification_dataset` folder.

3. **Install YOLOv5 Requirements:**
   ```bash
   cd yolov5
   pip install -r requirements.txt
   ```

4. **Install Additional Requirements for Pose Estimation:**
   ```bash
   pip install mediapipe
   ```

## Usage

### Training the YOLOv5 Model

1. **Train the YOLOv5 model:**
   ```bash
   python classify/train.py --model yolov5s-cls.pt --data ../data --epochs 20 --img 224 --batch 15
   ```

### Plotting Accuracy and Loss Graphs

1. **Plot accuracy and loss graphs from training results:**

   ```python
   import pandas as pd
   import matplotlib.pyplot as plt

   # Read the results.csv file
   df = pd.read_csv('runs/train-cls/exp/results.csv')

   # Extract accuracy and loss values
   train_acc = df['  metrics/accuracy_top1']
   train_loss = df['             train/loss']
   val_loss = df['               val/loss']

   # Plot the accuracy graph
   plt.plot(train_acc, label='Train')
   plt.title('Classification Accuracy')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.show()

   # Plot the loss graph
   plt.plot(train_loss, label='Train')
   plt.plot(val_loss, label='Validation')
   plt.title('Classification Loss')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend()
   plt.show()
   ```

### Predicting Sports

1. **Predict sports activities from a video using the trained model:**
   ```bash
   python classify/predict.py --weights runs/train-cls/exp/weights/best.pt --source ../Sports_Classification_dataset/cricket.mp4
   ```

### Pose Estimation

1. **Perform pose estimation and convert the output to a video:**

   ```python
   import cv2
   import mediapipe as mp
   import numpy as np

   mp_drawing = mp.solutions.drawing_utils
   mp_pose = mp.solutions.pose

   def calculate_angle(a, b, c):
       a = np.array(a) # First
       b = np.array(b) # Mid
       c = np.array(c) # Last

       radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
       angle = np.abs(radians * 180.0 / np.pi)
       
       if angle > 180.0:
           angle = 360 - angle
           
       return angle 

   cap = cv2.VideoCapture(r'yolov5/runs/predict-cls/exp/cricket.mp4')
   filename = "final.avi"
   codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
   width  = int(cap.get(3))
   height = int(cap.get(4))
   fps = 24
   resolution = (width, height)
   out_video = cv2.VideoWriter(filename, codec, fps, resolution) # To convert the frames to video

   with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
       while cap.isOpened():
           ret, image = cap.read()
           if not ret:
               break

           results = pose.process(image)

           try:
               landmarks = results.pose_landmarks.landmark

               # Get coordinates for various joints
               shoulderleft = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
               elbowleft = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
               wristleft = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
               shoulderright = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
               elbowright = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
               wristright = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
               hipleft = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
               kneeleft = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
               ankleleft = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
               hipright = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
               kneeright = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
               ankleright = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

               # Calculate angles
               angle1 = int(calculate_angle(shoulderleft, elbowleft, wristleft))
               angle2 = int(calculate_angle(shoulderright, elbowright, wristright))
               angle3 = int(calculate_angle(hipleft, kneeleft, ankleleft))
               angle4 = int(calculate_angle(hipright, kneeright, ankleright))
               
               # Print angles in the video
               cv2.putText(image, str(angle1), tuple(np.multiply(elbowleft, [width, height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
               cv2.putText(image, str(angle2), tuple(np.multiply(elbowright, [width, height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
               cv2.putText(image, str(angle3), tuple(np.multiply(kneeleft, [width, height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
               cv2.putText(image, str(angle4), tuple(np.multiply(kneeright, [width, height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
           except:
               pass

           mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,0,255), thickness=5, circle_radius=5), mp_drawing.DrawingSpec(color=(0,255,0), thickness=5, circle_radius=5))
           out_video.write(image)

   cv2.destroyAllWindows()
   out_video.release()
   cap.release()
   ```

## Code Structure

- **YOLOv5 Model Training:**
  - `yolov5/` - Directory with YOLOv5 code and requirements.
- **Scripts:**
  - `classify/train.py` - Training script for YOLOv5.
  - `classify/predict.py` - Prediction script for sports classification.
  - Pose Estimation Script - Performs pose estimation on the video.
