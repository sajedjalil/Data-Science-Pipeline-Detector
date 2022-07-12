import os
import gc
import sys
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping
from tensorflow.contrib.keras import backend

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from PIL import Image
from itertools import chain
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

def get_jpeg_data_files_paths():
    """
    Returns the input file folders path
    
    :return: list of strings
        The input file paths as list [train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file]
    """

    data_root_folder = os.path.abspath("../input/")
    train_jpeg_dir = os.path.join(data_root_folder, 'train-jpg')
    test_jpeg_dir = os.path.join(data_root_folder, 'test-jpg')
    test_jpeg_additional = os.path.join(data_root_folder, 'test-jpg-additional')
    train_csv_file = os.path.join(data_root_folder, 'train_v2.csv')
    return [train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file]

def preprocess_train_data(train_set_folder, train_csv_file, img_resize=(32, 32)):
    """
    Transform the train images to ready to use data for the CNN 
    :param train_set_folder: the folder containing the images for training
    :param train_csv_file: the file containing the labels of the training images
    :param img_resize: the standard size you want to have on images when transformed to matrices
    :param process_count: the number of process you want to use to preprocess the data.
        If you run into issues, lower this number. Its default value is equal to the number of core of your CPU
    :return: The images matrices and labels as [x_train, y_train, labels_map]
        x_train: The X train values as a numpy array
        y_train: The Y train values as a numpy array
        labels_map: The mapping between the tags labels and their indices
    """
    X_train, X_valid, y_train, y_valid, labels_map = _get_train_matrices(train_set_folder, train_csv_file, img_resize)
#    print("Done. Size consumed by arrays {} mb".format((ret[0].nbytes + ret[1].nbytes) / 1024 / 1024))
    return X_train, X_valid, y_train, y_valid, labels_map

def _get_train_matrices(train_set_folder, train_csv_file, img_resize):
    """
    :param train_set_folder: string
        The path of the all the train images
    :param train_csv_file: string
        The path of the csv file labels
    :param img_resize: tuple (int, int)
        The resize size of the original image given by the file_path argument
    :param process_count: int
        The number of threads you want to spawn to transform raw images to numpy
        matrices
    :return: x_train, y_train, labels_map
        x_train: list of float matrices
            The list of all the images stored as numpy matrices
        y_train: list of list of int
            A list containing vectors of 17 length long ints
        labels_map: dict {string: int}
            Inverted mapping of labels/id
    """
    labels_df = pd.read_csv(train_csv_file)
    labels = sorted(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))
    labels_map = {l: i for i, l in enumerate(labels)}

    x_train = []
    y_train = []

    inv_label_map = {i: l for l, i in labels_map.items()}
    
    for f, tags in tqdm(labels_df.values, miniters=1000):
        img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[labels_map[t]] = 1 
        x_train.append(cv2.resize(img, (64, 64)))
        y_train.append(targets)
        
    y_train = np.array(y_train, np.uint8)
    x_train = np.array(x_train, np.float16) / 255.
    
    print(x_train.shape)
    print(y_train.shape)

    split = 35000
    X_train, X_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

    return X_train, X_valid, y_train, y_valid, {v: ke for ke, v in labels_map.items()}

def creat_network(img_size=(32, 32), img_channels=3, output_size=17):
    classifier = Sequential()
    classifier.add(BatchNormalization(input_shape=(*img_size, img_channels)))

    classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=2))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=2))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=2))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    classifier.add(Conv2D(256, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=2))
    classifier.add(Dropout(0.25))
    
    classifier.add(Flatten())
    
    classifier.add(Dense(512, activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.5))
    classifier.add(Dense(output_size, activation='sigmoid'))
    
    return classifier

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
def train_model(classifier, X_train, X_valid, y_train, y_valid, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2):
    history = LossHistory()

    opt = Adam(lr=learn_rate)

    classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


    # early stopping will auto-stop training process if model stops learning after 3 epochs
    earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
    
    from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

    classifier.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epoch,
                        verbose=1,
                        validation_data=(X_valid, y_valid),
                        callbacks=[history, checkpoint, earlyStopping])
    fbeta_score = _get_fbeta_score(classifier, X_valid, y_valid)
    return [history.train_losses, history.val_losses, fbeta_score]

def _get_fbeta_score(classifier, X_valid, y_valid):
    p_valid = classifier.predict(X_valid)
    return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

######################################################################################################
# If the folders already exists then the files may already be extracted
# This is a bit hacky but it's sufficient for our needs
datasets_path = get_jpeg_data_files_paths()
for dir_path in datasets_path:
    if os.path.exists(dir_path):
        is_datasets_present = True
        
if not is_datasets_present:
    print("datasets are not present.")
else:
    print("All datasets are present.")

# ## Inspect image labels
# Visualize what the training set looks like
train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = get_jpeg_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
print(labels_df.head())

# Print all unique tags
labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_set = set(labels_list)
print("There is {} unique labels including {}".format(len(labels_set), labels_set))
    
# # Image Resize
# Define the dimensions of the image data trained by the network. Due to memory constraints we can't load in the full size 256x256 jpg images. Recommended resized images could be 32x32, 64x64, or 128x128.
img_resize = (64, 64) # The resize size of each image

# # Data preprocessing
# Preprocess the data in order to fit it into the Keras model.
# 
# Due to the hudge amount of memory the resulting matrices will take, the preprocessing will be splitted into several steps:
#     - Preprocess training data (images and labels) and train the neural net with it
#     - Delete the training data and call the gc to free up memory
#     - Preprocess the first testing set
#     - Predict the first testing set labels
#     - Delete the first testing set
#     - Preprocess the second testing set
#     - Predict the second testing set labels and append them to the first testing set
#     - Delete the second testing set
X_train, X_valid, y_train, y_valid, y_map = preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)
# Free up all available memory space after this heavy operation
gc.collect();
  
# ## Create a checkpoint
# 
# Creating a checkpoint saves the best model weights across all epochs in the training process. This ensures that we will always use only the best weights when making our predictions on the test set rather than using the default which takes the final score from the last epoch. 


# ## Choose Hyperparameters
# 
# Choose your hyperparameters below for training. 
validation_split_size = 0.2
batch_size = 128

# ## Define and Train model
# 
# Here we define the model and begin training. 
# 
# Note that we have created a learning rate annealing schedule with a series of learning rates as defined in the array `learn_rates` and corresponding number of epochs for each `epochs_arr`. Feel free to change these values if you like or just use the defaults. 
classifier = creat_network(img_resize, img_channels=3, output_size=len(y_map))

train_losses, val_losses = [], []
epochs_arr = [10, 5, 5]
learn_rates = [0.001, 0.0001, 0.00001]
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    tmp_train_losses, tmp_val_losses, fbeta_score_temp = train_model(classifier, X_train, X_valid, y_train, y_valid, learn_rate, epochs, 
                                                                           batch_size, validation_split_size=validation_split_size)
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses
    
# ## Load Best Weights
# Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training
classifier.load_weights("weights.best.hdf5")
print("Weights loaded")

# ## Monitor the results
# Check that we do not overfit by plotting the losses of the train and validation sets
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend();
          
# Look at our fbeta_score
print(fbeta_score_temp)