# Brain Tumor Detection using CNN

A deep learning project that detects the presence of brain tumors from MRI images using a Convolutional Neural Network (CNN).
The model is trained on labeled MRI scans and learns visual patterns that distinguish tumor and non-tumor brain images.

This project aims to demonstrate the application of **computer vision and deep learning in medical image analysis**, providing an automated approach that can assist in early tumor detection.

---

## Project Status

🚧 **Currently in development**

Planned components include:

* MRI image preprocessing
* CNN model training and evaluation
* Tumor prediction on new MRI images
* Visualization of training performance
* Model deployment for inference

---

## Features

* MRI image preprocessing (resizing, normalization)
* Binary classification: **Tumor / No Tumor**
* Convolutional Neural Network built using TensorFlow/Keras
* Model evaluation using accuracy and validation metrics
* Visualization of training performance
* Prediction on new MRI scans

---

## Tech Stack

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**

---

## Dataset

The model is trained using a publicly available **Brain MRI dataset** containing labeled MRI images categorized as:

* **Tumor**
* **No Tumor**

The dataset will be used to train and validate the CNN model.

---

## Project Structure

brain-tumor-detection-cnn
│
├── dataset/            # MRI dataset (tumor / no tumor images)
├── notebooks/          # Jupyter notebooks for experiments
├── src/                # Training and preprocessing scripts
├── models/             # Saved trained models
├── requirements.txt    # Project dependencies
└── README.md

---

## Model Architecture (Planned)

The CNN model will include:

* Convolutional layers for feature extraction
* Max pooling layers for dimensionality reduction
* Fully connected dense layers for classification
* Dropout layers to reduce overfitting

---

## Future Improvements

* Data augmentation for improved generalization
* Grad-CAM visualization for model interpretability
* Streamlit web interface for MRI upload and prediction
* Performance optimization and hyperparameter tuning

---

## How to Run (Coming Soon)

Instructions for running the project locally will be added once the training pipeline is completed.

---


