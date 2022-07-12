import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image

# Permission denied
# lib
# src


# pwd
# /kaggle/working

# .
# __output__.json

# ../
# input
# lib
# src
# working

# ../input
# sample_submission_v2.csv
# test-jpg-v2
# test-tif-v3
# test_v2_file_mapping.csv
# train-jpg
# train-tif-v2
# train_v2.csv



labels = {'haze', 'slash_burn', 'selective_logging', 'primary', 'artisinal_mine', 'bare_ground', 'blooming', 'partly_cloudy', 'road', 'water', 'habitation', 'blow_down', 'cultivation', 'conventional_mine', 'agriculture', 'cloudy', 'clear'}



#####
import numpy as np
import os

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping
from tensorflow.contrib.keras import backend


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class AmazonKerasClassifier:
    def __init__(self):
        self.losses = []
        self.classifier = Sequential()

    def add_conv_layer(self, img_size=(32, 32), img_channels=3):
        self.classifier.add(BatchNormalization(input_shape=(*img_size, img_channels)))

        self.classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(32, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(64, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(128, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(256, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))


    def add_flatten_layer(self):
        self.classifier.add(Flatten())


    def add_ann_layer(self, output_size):
        self.classifier.add(Dense(512, activation='relu'))
        self.classifier.add(BatchNormalization())
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(output_size, activation='sigmoid'))

    def _get_fbeta_score(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def train_model(self, x_train, y_train, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2, train_callbacks=()):
        history = LossHistory()

        X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                              test_size=validation_split_size)

        opt = Adam(lr=learn_rate)

        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

        self.classifier.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=1,
                            validation_data=(X_valid, y_valid),
                            callbacks=[history, *train_callbacks, earlyStopping])
        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def save_weights(self, weight_file_path):
        self.classifier.save_weights(weight_file_path)

    def load_weights(self, weight_file_path):
        self.classifier.load_weights(weight_file_path)

    def predict(self, x_test):
        predictions = self.classifier.predict(x_test)
        return predictions

    def map_predictions(self, predictions, labels_map, thresholds):
        """
        Return the predictions mapped to their labels
        :param predictions: the predictions from the predict() method
        :param labels_map: the map
        :param thresholds: The threshold of each class to be considered as existing or not existing
        :return: the predictions list mapped to their labels
        """
        predictions_labels = []
        for prediction in predictions:
            labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels

    def close(self):
        backend.clear_session()
#####

def _train_transform_to_matrices(*args):
    """
    
    :param args: list of arguments
        file_path: string
            The path of the image
        tags: list of strings
            The associated tags
        labels_map: dict {int: string}
            The map between the image label and their id 
        img_resize: tuple (int, int)
            The resize size of the original image given by the file_path argument
    :return: img_array, targets
        img_array: Numpy array
            The image from the file_path as a numpy array resized with img_resize
        targets: Numpy array
            A 17 length vector
    """
    # Unpack the *args
    file_path, tags, labels_map, img_resize = list(args[0])
    img = Image.open(file_path)
    img.thumbnail(img_resize)  # Resize the image

    # Augment the image `img` here

    # Convert to RGB and normalize
    img_array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255

    targets = np.zeros(len(labels_map))
    for t in tags.split(' '):
        targets[labels_map[t]] = 1
    return img_array, targets


def _get_train_matrices(train_set_folder, train_csv_file, img_resize, process_count):
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

    files_path = []
    tags_list = []
    for file_name, tags in labels_df.values:
        files_path.append('{}/{}.jpg'.format(train_set_folder, file_name))
        tags_list.append(tags)

    x_train = []
    y_train = []
    # Multiprocess transformation, the map() function take a function as a 1st argument
    # and the argument to pass to it as the 2nd argument. These arguments are processed
    # asynchronously on threads defined by process_count and their results are stored in
    # the x_train and y_train lists
    with ThreadPoolExecutor(process_count) as pool:
        for img_array, targets in tqdm(pool.map(_train_transform_to_matrices,
                                                [(file_path, tag, labels_map, img_resize)
                                                 for file_path, tag in zip(files_path, tags_list)]),
                                       total=len(files_path)):
            x_train.append(img_array)
            y_train.append(targets)
    return [x_train, y_train, {v: k for k, v in labels_map.items()}]


def preprocess_train_data(train_set_folder, train_csv_file, img_resize=(32, 32), process_count=cpu_count()):
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
    x_train, y_train, labels_map = _get_train_matrices(train_set_folder, train_csv_file, img_resize, process_count)
    ret = [np.array(x_train), np.array(y_train, dtype=np.uint8), labels_map]
    print("Done. Size consumed by arrays {} mb".format((ret[0].nbytes + ret[1].nbytes) / 1024 / 1024))
    return ret

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


train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = get_jpeg_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
labels_df.head()

from itertools import chain
labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_set = set(labels_list)
print("There is {} unique labels including {}".format(len(labels_set), labels_set))

# Histogram of label instances
labels_s = pd.Series(labels_list).value_counts() # To sort them by count
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=labels_s, y=labels_s.index, orient='h')


images_title = [labels_df[labels_df['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg' 
                for i, label in enumerate(labels_set)]

images_title = images_title[1:100]

plt.rc('axes', grid=False)
_, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))
axs = axs.ravel()

for i, (image_name, label) in enumerate(zip(images_title, labels_set)):
    img = mpimg.imread(train_jpeg_dir + '/' + image_name)
    axs[i].imshow(img)
    axs[i].set_title('{} - {}'.format(image_name, label))

img_resize = (64, 64) # The resize size of each image
validation_split_size = 0.2
batch_size = 128


x_train, y_train, y_map = preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)
# Free up all available memory space after this heavy operation
gc.collect();

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
y_map


from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

classifier = AmazonKerasClassifier()
classifier.add_conv_layer(img_resize)
classifier.add_flatten_layer()
classifier.add_ann_layer(len(y_map))

train_losses, val_losses = [], []
epochs_arr = [20, 5, 5]
learn_rates = [0.001, 0.0001, 0.00001]
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model(x_train, y_train, learn_rate, epochs, 
                                                                           batch_size, validation_split_size=validation_split_size, 
                                                                           train_callbacks=[checkpoint])
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses
    
classifier.load_weights("weights.best.hdf5")
print("Weights loaded")

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend();


fbeta_score

del x_train, y_train
gc.collect()


x_test, x_test_filename = data_helper.preprocess_test_data(test_jpeg_dir, img_resize)
# Predict the labels of our x_test images
predictions = classifier.predict(x_test)

del x_test
gc.collect()

x_test, x_test_filename_additional = data_helper.preprocess_test_data(test_jpeg_additional, img_resize)
new_predictions = classifier.predict(x_test)

del x_test
gc.collect()
predictions = np.vstack((predictions, new_predictions))
x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape, 
                                                                              x_test_filename.shape,
                                                                              predictions[0]))
                                                                              
# For now we'll just put all thresholds to 0.2 
thresholds = [0.2] * len(labels_set)

# TODO complete
tags_pred = np.array(predictions).T
_, axs = plt.subplots(5, 4, figsize=(15, 20))
axs = axs.ravel()

for i, tag_vals in enumerate(tags_pred):
    sns.boxplot(tag_vals, orient='v', palette='Set2', ax=axs[i]).set_title(y_map[i])

predicted_labels = classifier.map_predictions(predictions, y_map, thresholds)

tags_list = [None] * len(predicted_labels)
for i, tags in enumerate(predicted_labels):
    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]

final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
final_df.head()

tags_s = pd.Series(list(chain.from_iterable(predicted_labels))).value_counts()
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=tags_s, y=tags_s.index, orient='h');

final_df.to_csv('../submission_file.csv', index=False)
classifier.close()

