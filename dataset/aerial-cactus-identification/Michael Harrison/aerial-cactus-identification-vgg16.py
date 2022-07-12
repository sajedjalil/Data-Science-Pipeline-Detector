import tensorflow as tf
tf.enable_eager_execution()

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


### Load Data
train_df = pd.read_csv("../input/train.csv")
image_paths = "../input/train/train/" + train_df.id.values
image_labels = train_df.has_cactus.values

# Test Data
dir_test = "../input/test/test/"
for root, dirs, files in os.walk(dir_test):
    pass

image_paths_test = np.array([dir_test + file for file in files])
num_images_test = len(image_paths_test)

# Create dummy test labels (needed to create tensorflow dataset)
image_labels_test = np.array([0] * num_images_test)


### Data Checks

# Check image dimensions - expect all to be 32x32
for img_path in image_paths:
    img = Image.open(img_path)
    if img.size != (32, 32):
        print("IMAGES DIMENSION CHECK FAILED!")
        exit()

# Do the same for the test images
for img_path in image_paths_test:
    img = Image.open(img_path)
    if img.size != (32, 32):
        print("IMAGES DIMENSION CHECK FAILED!")
        exit()

# Check training labels - should all be 0 or 1 
labels_check = np.in1d(image_labels, (0,1)) # tests if each element of image_labels is in (0,1)
if not np.all(labels_check): # i.e. at least one False in labels_check
    print("LABELS CHECK FAILED!")
    exit()
    

### Split Data into Training & Validation sets - 90:10 split
(image_paths_train, image_paths_valid, image_labels_train, image_labels_valid) = train_test_split(image_paths, \
    image_labels, \
    test_size = 0.1, \
    random_state = 1905)

num_images_train = image_paths_train.shape[0]
num_images_valid = image_paths_valid.shape[0]


### Training Params
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_STEPS_PER_EPOCH_TRAIN = int(np.floor(num_images_train/BATCH_SIZE))  # Use entire dataset per epoch; round up to ensure entire dataset is covered if batch_size does not divide into num_images
NUM_STEPS_PER_EPOCH_VALID = int(np.floor(num_images_valid/BATCH_SIZE))
SHUFFLE_SEED = 1905


### Create Tensorflow Datasets
def read_image(filename, label, num_channels):
    """
    Utility function to take the given image path into a tensorflow tensor, and then standardise its values
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_image(image_string, channels = num_channels) # Represents images as (height, width, channels)
    image /= 255 # Normalise pixel values
    return image, label

def create_dataset(filenames, labels, num_epochs, batch_size, shuffle_seed, shuffle = True):
    """
    Utility function to load a dataset from a given set of matching filenames & labels, then shuffles and batches the dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(lambda filename, label: read_image(filename, label, num_channels = 3))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size = len(filenames), \
            reshuffle_each_iteration = True, \
            seed = shuffle_seed)
            
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return(dataset)


dataset_train = create_dataset(image_paths_train, \
    image_labels_train, \
    num_epochs = NUM_EPOCHS, \
    batch_size = BATCH_SIZE, \
    shuffle_seed = SHUFFLE_SEED)

dataset_valid = create_dataset(image_paths_valid, \
    image_labels_valid, \
    num_epochs = NUM_EPOCHS, \
    batch_size = BATCH_SIZE, \
    shuffle_seed = SHUFFLE_SEED)


### Create Model - use pre-trained VGG16 as convolutional base
vgg16_base = VGG16(weights = 'imagenet', include_top = False, pooling = None)
vgg16_base.trainable = False

# Attach Input
image_input = Input(shape = (32, 32, 3))
x = vgg16_base(image_input)

# Attach new output "head" - whose weights will be trained
x = Flatten()(x)
x =  Dense(512, activation = 'relu')(x)
out =  Dense(1, activation = 'sigmoid')(x)

model = Model(image_input, out)
print(model.summary())


### TRAINING

# Compile Model
model.compile(optimizer = Adam(), \
    loss='binary_crossentropy', \
    metrics=['accuracy'])

# Train Model
model.fit(dataset_train, \
epochs = NUM_EPOCHS, \
steps_per_epoch = NUM_STEPS_PER_EPOCH_TRAIN, \
validation_data = dataset_valid, \
validation_steps = NUM_STEPS_PER_EPOCH_VALID)

# Plot Training History
model_history = model.history.history

# Accuracy
plt.plot(model_history['acc'])
plt.plot(model_history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig("accuracy.jpg")
plt.clf()

# Loss:
plt.plot(model_history['loss'])
plt.plot(model_history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig("loss.jpg")
plt.clf()


### Test Predictions

# Create test dataset
dataset_test = create_dataset(image_paths_test, \
    image_labels_test, \
    num_epochs = 1, \
    batch_size = num_images_test, \
    shuffle = False, \
    shuffle_seed = None)

# Run test dataset through the model and put results into a dataframe
test_predictions = model.predict(dataset_test, steps = 1)
test_output = {"id": files, "has_cactus": test_predictions.flatten()}
test_output = pd.DataFrame(test_output)


#### Save Output
model.save_weights('model_file.h5')
test_output.to_csv("submission.csv", index = False)
