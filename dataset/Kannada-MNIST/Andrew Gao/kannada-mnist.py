#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          LeakyReLU, MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

MODEL_FILE = "model.h5"
MODEL_SUMMARY_FILE = "model_summary.txt"
KAGGLE_SUBMISSION_FILE = "submission.csv"
ACCURACY_FILE = "accuracy.png"
LOSS_FILE = "loss.png"

IMAGE_SIZE = 28
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
CHANNELS = 1
CLASSES = 10
VALIDATION_RATIO = 0.2
BATCH_SIZE = 1024
EPOCHS = 53
VERBOSITY = 1

# Extract data
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

y = train["label"]
x = train.drop(labels=["label"], axis=1)
test = test.drop(labels=["id"], axis=1)

# Reshape data
x = x.values.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
test = test.values.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)

# One-hot encoding
y = to_categorical(y, num_classes=CLASSES)

# Prepare training / validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=VALIDATION_RATIO, random_state=42, shuffle=True
)

# Build model
model = Sequential(
    [
        Conv2D(64, (3, 3), padding="same", input_shape=(28, 28, 1)),
        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        LeakyReLU(alpha=0.1),
        Conv2D(64, (3, 3), padding="same"),
        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        LeakyReLU(alpha=0.1),
        Conv2D(64, (3, 3), padding="same"),
        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        LeakyReLU(alpha=0.1),
        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        LeakyReLU(alpha=0.1),
        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        Conv2D(256, (3, 3), padding="same"),
        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        LeakyReLU(alpha=0.1),
        Conv2D(256, (3, 3), padding="same"),
        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        Flatten(),
        Dense(256),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=RMSprop(learning_rate=0.001, rho=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Save model summary
with open(MODEL_SUMMARY_FILE, "w") as fn:
    model.summary(print_fn=lambda line: fn.write(line + "\n"))

# Data augmentation
data_generator = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    shear_range=5,
    zoom_range=0.15,
    horizontal_flip=False,
)
data_generator.fit(x_train)

# Reduce learning rate when accuracy has stopped improving
learning_rate_reduction = ReduceLROnPlateau(
    monitor="accuracy",
    patience=5,
    mode="auto",
    factor=0.1,
    min_delta=0.0001,
    min_lr=0.00001,
    verbose=VERBOSITY,
)

# Stop training when validation loss has stopped improving
es = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=300, restore_best_weights=True
)

# Train model
history = model.fit_generator(
    data_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
    callbacks=[learning_rate_reduction, es],
    verbose=VERBOSITY,
)

# Save model weights
model.save_weights(MODEL_FILE)

# Predict classes
predictions = model.predict_classes(test, verbose=1)
pd.DataFrame({"id": list(range(0, len(predictions))), "label": predictions}).to_csv(
    KAGGLE_SUBMISSION_FILE, index=False
)

# Plot training / validation accuracy values
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="lower right")
plt.show()
plt.savefig(ACCURACY_FILE)

# Plot training / validation loss values
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper right")
plt.show()
plt.savefig(LOSS_FILE)