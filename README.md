# Facial Expression Driven Emotion Recognition Using a Custom Deep CNN Architecture
## Overview

This project uses a Custom Convolutional Neural Network (CNN) implemented in MATLAB to recognize human emotions from facial images. The model classifies facial expressions into seven categories: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

## Features
-Facial Emotion Recognition
-Custom CNN Architecture
-Training, Validation, and Testing Pipeline
-Confusion Matrix Visualization
-Deep Learning-Based Classification
-MATLAB Implementation
-Dataset

This project uses the FER-2013 facial emotion recognition dataset obtained from Kaggle.

## Original Dataset:
https://www.kaggle.com/datasets/msambare/fer2013

## Project Dataset:
A modified version of the FER-2013 dataset was used. Some images were removed during preprocessing to reduce dataset size and training time while maintaining representation across emotion classes.

### Emotion Classes
-Angry
-Disgusted
-Fearful
-Happy
-Neutral
-Sad
-Surprised
## Technologies Used
--MATLAB
--Deep Learning Toolbox
--Convolutional Neural Networks (CNN)
--Image Processing


### CNN Architecture
-Input Layer (48×48)
-Convolution Layer
-Batch Normalization
-ReLU Activation
-Max Pooling
-Fully Connected Layer
-Softmax Layer
-Classification Layer


## Project Workflow
-Dataset Collection
-Data Preprocessing
-Train-Validation-Test Split
-CNN Model Design
-Model Training
-Model Evaluation
-Emotion Prediction

## Results
-Training Accuracy   = 94-97%
-Validation Accuracy = 85-89%
-Testing Accuracy    = 84-88%


## How to Run
Download the dataset.
Place the dataset folder in the MATLAB project directory.
Open MATLAB.
Run the main script.
View training progress and confusion matrix.


## Future Improvements
Dataset Balancing
Data Augmentation
Transfer Learning (ResNet-18, GoogLeNet)
Improved model accuracy
Mobile application integration
Real-Time Webcam-Based Emotion Detection


## Author

Md Thanveer Jaha
