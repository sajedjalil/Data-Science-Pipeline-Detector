# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time, gc
import tensorflow as tf
from PIL import Image
print(tf.__version__)

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

# import the necessary keras and sklearn packages

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

# %% [markdown]
# 

# %% [code]
print(train_df_.head())

# %% [code]
len(train_df_)

# %% [code]
print(class_map_df.head())

# %% [code]
print(class_map_df.component_type.value_counts())

# %% [code]
class_map_df_root = class_map_df[class_map_df.component_type=='grapheme_root']
class_map_df_vowel = class_map_df[class_map_df.component_type=='vowel_diacritic']
class_map_df_cons = class_map_df[class_map_df.component_type=='consonant_diacritic']

# %% [code]
graphemeLB = LabelBinarizer()
vowelLB = LabelBinarizer()
consonantLB = LabelBinarizer()

graphemeLB.fit(class_map_df_root.label)
vowelLB.fit(class_map_df_vowel.label)
consonantLB.fit(class_map_df_cons.label)

# %% [code]
print(len(vowelLB.classes_))
print(len(consonantLB.classes_))
print(len(graphemeLB.classes_))

# %% [code]
def read_data(nf):
    nf=int(nf)
    train_df = pd.read_feather(f'/kaggle/input/bengaliaicv19feather/train_image_data_{nf}.feather')
    return train_df


# initialize the optimizer and compile the model
print("[INFO] loading saved models from running on first 2 datasets...")
model_root = tf.keras.models.load_model('/kaggle/input/bengali-graphemes-resnet50/model_root.h5')
model_vowel = tf.keras.models.load_model('/kaggle/input/bengali-graphemes-resnet50/model_vowel.h5')
model_consonant = tf.keras.models.load_model('/kaggle/input/bengali-graphemes-resnet50/model_consonant.h5')

# %% [code]
EPOCHS = 15
BS = 128

# %% [markdown]
# ## Read image data from feather format, binarize the labels, train on full data, create TF Dataset from images and labels, and do model.fit in a loop for the 4 sets of images

# %% [code]
histories = []

es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

for i in range(2,4):
    print("iteration:"+str(i))

    graphemeLabels = []
    vowelLabels = []
    consonantLabels = []   
    print("[INFO] reading train images and labels...")
    train_df = pd.merge(read_data(i), train_df_, on='image_id').drop(['image_id','grapheme'], axis=1)
    train_df = train_df.astype('uint8')
    graphemeLabels = train_df.grapheme_root
    vowelLabels = train_df.vowel_diacritic
    consonantLabels = train_df.consonant_diacritic

    print("[INFO] binarizing labels...")
    graphemeLabels = graphemeLB.transform(np.array(graphemeLabels))
    vowelLabels = vowelLB.transform(np.array(vowelLabels))
    consonantLabels = consonantLB.transform(np.array(consonantLabels))

    print(graphemeLabels.shape)
    print(vowelLabels.shape)
    print(consonantLabels.shape)

    train_df=train_df.drop(["consonant_diacritic","grapheme_root","vowel_diacritic"],axis=1)
    
    print("[INFO] doing train test split...")
    (trainX, testX, trainGraphemeY, testGraphemeY,trainVowelY, testVowelY,trainConsonantY,testConsonantY) = train_test_split(train_df, graphemeLabels, vowelLabels,consonantLabels,test_size=0.01, random_state=42)
   
    del train_df
    del graphemeLabels
    del vowelLabels
    del consonantLabels
    gc.collect()

    print("[INFO] creating train dataset...")
    trainX=np.array(trainX).reshape(-1,137,236,1)
    print(trainX.shape)
    resized_image=[]
    for j in range(trainX.shape[0]):
        resized_img = tf.image.resize(trainX[j],[96,96])
        resized_img=np.array(resized_img)/255.
        resized_image.append(resized_img)
    resized_image = np.asarray(resized_image)

    del trainX
    gc.collect()
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    datagen.fit(resized_image)
    
    
    print("[INFO] creating validation dataset...")
    testX=np.array(testX).reshape(-1,137,236,1)
    print(testX.shape)
    resized_image_test=[]
    for i in range(len(testX)):
        resized_img = tf.image.resize(testX[i],[96,96])
        resized_img=np.array(resized_img)/255.
        resized_image_test.append(resized_img)
    resized_image_test = np.asarray(resized_image_test)

    del testX
    gc.collect()

    print("[INFO] Root Model.fit starting...")

    history = model_root.fit_generator(datagen.flow(resized_image, trainGraphemeY, batch_size=BS),
                                      epochs = EPOCHS, validation_data = (resized_image_test,testGraphemeY),
                                      steps_per_epoch=resized_image.shape[0] // BS, 
                                      callbacks=[es],verbose=2)

    histories.append(history)
    
    print("[INFO] Vowel Model.fit starting...")
    history = model_vowel.fit_generator(datagen.flow(resized_image, trainVowelY, batch_size=BS),
                                      epochs = EPOCHS, validation_data = (resized_image_test,testVowelY),
                                      steps_per_epoch=resized_image.shape[0] // BS, 
                                      callbacks=[es],verbose=2)

    histories.append(history)
    
    print("[INFO] Cons Model.fit starting...")
    history = model_consonant.fit_generator(datagen.flow(resized_image, trainConsonantY, batch_size=BS),
                                      epochs = EPOCHS, validation_data = (resized_image_test,testConsonantY),
                                      steps_per_epoch=resized_image.shape[0] // BS, 
                                      callbacks=[es],verbose=2)

    histories.append(history)
    
    del resized_image
    del resized_image_test
    gc.collect()

# %% [code]
print("[INFO] Now saving Models")
model_root.save('model_root_1.h5')
model_vowel.save('model_vowel_1.h5')
model_consonant.save('model_consonant_1.h5')

# %% [code]
def plot_loss(his, epoch, i, title):
    plt.style.use('ggplot')
    fig=plt.figure()
    
    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')
    plt.plot(np.arange(0, epoch), his.history['val_loss'], label='val_loss')
  
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    fig.savefig('plot_loss'+str(i)+'.png')

def plot_acc(his, epoch, i, title):
    plt.style.use('ggplot')
    fig=plt.figure()
    plt.plot(np.arange(0, epoch), his.history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, epoch), his.history['val_accuracy'], label='val_accuracy')
     
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()
    fig.savefig('plot_acc'+str(i)+'.png')

# %% [code]
for dataset in range(len(histories)):
    plot_loss(histories[dataset], EPOCHS, dataset, f'Training Dataset: {dataset}')
    plot_acc(histories[dataset], EPOCHS, dataset, f'Training Dataset: {dataset}')