'''
Whale Tails

This is my first image classification challenge.
Some of the workflow was borrowed from section 5.2 of
Machine Learning with Python by François Chollet.
(The dogs vs cats example)

Obviously the dogs vs cats was binary classification,
and this is multiple class.

The training data were split into two relatively equal parts: training & validation.
The new_whale class was ignored.

I kept the images as large as I could and still have them fit into
the 6GB RAM of my GPU. Each epoch completes in about 6 minutes.

So far this only runs as far as validation.
After 20 epochs, val_acc converges to 0.02,
Given that there are 1902 unique classes (not counting new_whale),
random performance would yield val_acc of 0.0005.

So yes, this beats the baseline. But perhaps it could perform much better
with tweaking.

If you have any tips for a newb, do chime in.
'''


from collections import OrderedDict
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import csv
import numpy as np
import os
import pathlib
import pdb
import random
import shutil
import subprocess


# This class is used for image deduplication,
# since we don't want the same image in our test set
# and our validation set
class PhotoAlbum:
    def __init__(self, *paths):
        self.paths = paths
        self.used_shas = set()
        self.unique_images = set()
        self.image_map = {}

    def process(self):
        joined_paths = ' '.join(self.paths)
        cmd = f'sha256sum { joined_paths }'
        process = subprocess.Popen(cmd,
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        for line in process.stdout:
            sha, image = line.decode('utf-8').split()
            self.image_map[image] = sha
            if sha not in self.used_shas:
                self.used_shas.add(sha)
                self.unique_images.add(image)

    def __contains__(self, image):
        return(image in self.unique_images)




# Keras intuits targets from the name of the subdirectory.
# For example:
#   train/
#       target_1/
#           image_p.jpg
#           image_x.jpg
#           image_y.jpg
#           image_z.jpg
#       target_2/
#           image_a.jpg
#           image_b.jpg
#           image_c.jpg
#
#   validation/
#       target_1/
#           image_u.jpg
#           image_w.jpg
#       target_2/
#           image_d.jpg
#           image_e.jpg
#           image_f.jpg


def copy_image_to_subdir(image, label, src_dir, dest_dir):
    if label == 'new_whale':
        # Do not train on new_whale
        return
    if 'train' in dest_dir:
        num_images_in_use['train'] += 1
    elif 'validation' in dest_dir:
        num_images_in_use['validation'] += 1
    else:
        raise(ValueError)
    labels_in_use.add(label)
    subdir = f'{dest_dir}/{label}'
    pathlib.Path(subdir).mkdir(parents=True, exist_ok=True)
    shutil.copy(f'{src_dir}/{image}', subdir)



#input_dir = 'dataset'
input_dir = '../input'

# These are populated via ./prepare_data.sh
train_source_dir = f'{input_dir}/train'
test_source_dir  = f'{input_dir}/test'

# These are created and populated later in this file
train_dir      = 'dataset/samples/train'
validation_dir = 'dataset/samples/validation'

# This is populated as images are copied to subdirectories
# -- Only labels containing two or more original training images will be used --
labels_in_use = set()
num_images_in_use = dict(train=0, validation=0)



album = PhotoAlbum(f'{input_dir}/train/*', f'{input_dir}/test/*')
album.process()


# This is the source used to populate the subdirectories
# where each subdirectory represents a different label
# { 'some_image.jpg': 'some_label', ...}
train_source_dict = dict()
with open(f'{input_dir}/train.csv') as file:
    for row in csv.DictReader(file):
        image = row['Image']
        if f'{input_dir}/train/{image}' in album:
            train_source_dict[image] = row['Id']





# Invert train_source_dict so we can put all images
# of the same label in the same subdirectory
# { 'some_label': set(['some_image_1', 'some_image_2']) }
invert_train_source_dict = dict()
for image, label in train_source_dict.items():
    if not label in invert_train_source_dict:
        invert_train_source_dict[label] = set()
    invert_train_source_dict[label].add(image)



# Create subdirs and copy images
# IFF THERE ARE AT LEAST TWO IMAGES for this label
for label, images in invert_train_source_dict.items():
    num_images = len(images)
    midpoint = num_images // 2
    if num_images >= 2:
        image_list = list(images)
        random.shuffle(image_list)
        # If there are an odd number of keys, the extra one goes in TRAIN
        # Since we need extra train images more than we need validation images
        train_images      = image_list[-midpoint:]
        validation_images = image_list[:-midpoint]
        for image in train_images:
            copy_image_to_subdir(image, label, train_source_dir, train_dir)
        for image in validation_images:
            copy_image_to_subdir(image, label, train_source_dir, validation_dir)



# Step 1 scales image values and optionally adds variations
train_g      = ImageDataGenerator(rescale=1./255)
validation_g = ImageDataGenerator(rescale=1./255)


# Step 2 flows images from specified directory in specified batch sizes
train_generator = train_g.flow_from_directory(
        train_dir,
        target_size=(400, 800), # This is as large as I can go with 6GB of RAM
        batch_size=1,           # This is as large as I can go with 6GB of RAM
        class_mode='categorical')


validation_generator = validation_g.flow_from_directory(
        validation_dir,
        target_size=(400, 800),
        batch_size=1,          # Long delays happen if this is greater than 1
        class_mode='categorical')


num_labels_in_use = len(labels_in_use)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',
            input_shape=(400, 800, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))  # TODO should this be bigger?

# Shape of final layer must be same as number of distinct training labels
model.add(layers.Dense(num_labels_in_use,   # dynanically generated ()
                       activation='softmax')) # activation from table pp 114

model.compile(loss='categorical_crossentropy',   # loss from table pp 114
              optimizer=optimizers.RMSprop(lr=1e-4), # optimizer used in dogs/cats pp 13
              metrics=['acc'])


# Here is the model.summary()
#
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 398, 798, 32)      896
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 199, 399, 32)      0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 197, 397, 64)      18496
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 98, 198, 64)       0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 96, 196, 128)      73856
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 48, 98, 128)       0
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 46, 96, 256)       295168
# _________________________________________________________________
# max_pooling2d_4 (MaxPooling2 (None, 23, 48, 256)       0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 282624)            0
# _________________________________________________________________
# dense_1 (Dense)              (None, 512)               144704000
# _________________________________________________________________
# dense_2 (Dense)              (None, 1902)              975726
# =================================================================
# Total params: 146,068,142
# Trainable params: 146,068,142
# Non-trainable params: 0
# _________________________________________________________________


history = model.fit_generator(
                train_generator,
                steps_per_epoch=num_images_in_use['train'],  # number of unique train images.
                epochs=1, # Only running 1 epoch on kaggle, because of 60-minut time limit
                validation_data=validation_generator,
                validation_steps=num_images_in_use['validation']) # number of unique validation images

model.save('two-or-more__no-new-whales__400x800.h5')

