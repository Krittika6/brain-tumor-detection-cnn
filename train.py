import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import precision_recall_curve
dataset_path = "dataset"

categories = ["tumor", "notumor"]

data = []
labels = []

for category in categories:

    path = os.path.join(dataset_path, category)

    label = categories.index(category)

    for img in os.listdir(path):

        img_path = os.path.join(path, img)

        image = cv2.imread(img_path)

        image = cv2.resize(image, (128,128))

        data.append(image)
        labels.append(label)

print("Images loaded:", len(data))
data = np.array(data)
labels = np.array(labels)

# normalize pixel values (0–255 → 0–1)
data = data / 255.0

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Training images:", X_train.shape[0])
print("Testing images:", X_test.shape[0])
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=32
)
loss, accuracy = model.evaluate(X_test, y_test)
y_probs = model.predict(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
print("Test Accuracy:", accuracy)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

print("Best threshold:", best_threshold)
model.save("model/brain_tumor_cnn.h5")