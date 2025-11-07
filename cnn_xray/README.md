# X-Ray Cardiomegaly Detection System

This project focuses on detecting cardiomegaly (enlargement of the heart) from chest X-ray images using deep learning.  
The model is built using a Convolutional Neural Network (CNN) trained on labeled X-ray data to classify images as either **Normal** or **Cardiomegaly**.

## Overview

Cardiomegaly is a condition where the heart is enlarged, which can be detected from chest X-ray images.  
The goal of this project is to provide an automated AI-based tool that helps in the early detection of cardiomegaly using computer vision and deep learning techniques.

## Model Description

The model used in this project is a custom **CNN (Convolutional Neural Network)** designed for binary image classification.  
It was trained on chest X-ray datasets and optimized to distinguish between normal heart size and cardiomegaly cases.

### Architecture Summary

- **Input:** 128x128 RGB chest X-ray images  
- **Convolutional Layers:** Multiple Conv2D layers with ReLU activation  
- **Pooling Layers:** MaxPooling2D layers for dimensionality reduction  
- **Fully Connected Layers:** Dense layers for classification  
- **Dropout Layer:** Used to prevent overfitting  
- **Output Layer:** Single neuron with sigmoid activation for binary prediction  

### Training Details

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy  
- **Batch Size:** 32  
- **Epochs:** Tuned experimentally  
- **Framework:** TensorFlow / Keras  

## Dataset

The model was trained using a chest X-ray dataset containing labeled images for both **Normal** and **Cardiomegaly** classes.  
The images were preprocessed by resizing to 128x128 pixels, normalizing pixel values, and converting grayscale images to RGB where needed.

## Application Overview

A simple Streamlit-based web interface was developed to make the model easily accessible.  
The user can upload an X-ray image, and the system will display the classification result along with the prediction confidence.

### Prediction Results

- **Normal:** No signs of cardiomegaly detected.  
- **Cardiomegaly Detected:** Signs of heart enlargement present.  
  (A medical consultation is advised for confirmation.)

## Purpose

This system was developed for academic and research purposes to explore the potential of deep learning in medical imaging.  
It demonstrates how CNNs can assist in medical image classification tasks and highlights the importance of AI-assisted diagnosis.

## Limitations

- The model should not be used as a standalone medical diagnostic tool.  
- It may produce false positives or negatives depending on image quality and dataset limitations.  
- Further validation on larger and more diverse datasets is recommended.

## Author

Developed as part of a research and learning project in medical image analysis using deep learning.
