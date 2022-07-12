#############
# Libraries #
#############

# Basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Images
import os
from PIL import Image

# Tensorflow
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical




########
# Data #
########

def get_pixel_data(filepath):
    """
    Get the pixel data from an image as a numpy array.
    """
    # Open the file
    image = Image.open(filepath)
    
    # Get the data
    pixel_data = np.array(image.getdata())
    pixel_data = pixel_data.reshape((32,32,3))
    
    # Close the file
    image.close()
    
    return pixel_data

# Train data
path = "../input/train/train/"
files = sorted(os.listdir(path))
train = np.zeros(shape = (len(files),32,32,3))

for i in range(len(files)):
    train[i,:,:,:] = get_pixel_data(path + files[i])
    
labels_train = pd.read_csv("../input/train.csv").sort_values("id")

# Test data
path = "../input/test/test/"
files = sorted(os.listdir(path))
test = np.zeros(shape = (len(files),32,32,3))
test_id = []
for i in range(len(files)):
    test[i,:,:,:] = get_pixel_data(path + files[i])

# Normalize the data
X_train = train / 255
y_train = labels_train["has_cactus"]
X_test  = test / 255 




#########
# Model #
#########

# Create the model
model = Sequential()
model.add(Conv2D(filters = 8, kernel_size = 3, activation = "relu", input_shape = (32, 32, 3)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 16, kernel_size = 3, activation = "relu"))
model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(2,  activation = "softmax"))
model.summary()

model.compile(optimizer = "adam",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])
model.fit(X_train, to_categorical(y_train), epochs = 10)

# Make the predictions
preds = model.predict_classes(X_test)
print("Label 0 (False): {}".format(np.sum(preds == 0)))
print("Label 1 (True):  {}".format(np.sum(preds == 1)))

# Save the results
results = pd.DataFrame({"id" : files, "has_cactus": preds})
results.to_csv("submission.csv", index = False)