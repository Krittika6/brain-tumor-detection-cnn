import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model("model/brain_tumor_cnn.h5")
img = cv2.imread("test2.jpg")
img = cv2.resize(img, (128,128))
img = img / 255.0
img = np.reshape(img, (1,128,128,3))
prediction = model.predict(img)
if prediction > 0.53916005:
    print("Tumor detected")
else:
    print("No tumor detected")