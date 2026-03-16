# Brain Tumor Detection using CNN

## Overview

This project implements a **Convolutional Neural Network (CNN) from scratch** to classify brain MRI scans as **tumor** or **non-tumor**.

The model is trained on a dataset of MRI images and achieves **~97% accuracy** on the test set.

---

## Features

* MRI image preprocessing using **OpenCV**
* CNN architecture built using **TensorFlow/Keras**
* Training and validation accuracy visualization
* Precision–Recall analysis for threshold tuning
* Prediction script for testing new MRI scans
* Model saving and reuse

---

## Project Structure

```
brain-tumor-detection-cnn
│
├── dataset/              # MRI dataset (not uploaded to repo)
│   ├── tumor
│   └── notumor
│
├── model/
│   └── brain_tumor_cnn.h5
│
├── train.py              # Model training script
├── predict.py            # Predict tumor from MRI image
├── notebook/             # Experiments / analysis
└── README.md
```

---

## Model Architecture

CNN built from scratch:

```
Input Image (128x128x3)
↓
Conv2D (32 filters)
↓
MaxPooling
↓
Conv2D (64 filters)
↓
MaxPooling
↓
Conv2D (128 filters)
↓
MaxPooling
↓
Flatten
↓
Dense (128)
↓
Dropout (0.5)
↓
Output (Sigmoid)
```

---

## Dataset

Brain MRI dataset containing tumor and non-tumor scans.

Classes:

* Tumor
* No Tumor

Images are resized to **128×128** and normalized before training.

---

## Training

Run the training script:

```
python train.py
```

Training includes:

* Data preprocessing
* Train/test split (80/20)
* CNN training
* Accuracy and loss visualization
* Model saving

---

## Results

Test Accuracy:

```
~97.8%
```

The precision–recall analysis is used to determine the optimal prediction threshold.

---

## Predicting on a New MRI Scan

Place an MRI image in the project folder and run:

```
python predict.py
```

Example output:

```
Tumor detected
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib

---

## Future Improvements

* Grad-CAM visualization for tumor localization
* Web interface for MRI upload and prediction
* Model deployment as a web application
