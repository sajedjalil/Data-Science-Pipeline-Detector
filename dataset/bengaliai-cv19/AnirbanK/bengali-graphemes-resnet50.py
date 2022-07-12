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

# %% [code]
def res_net_block_1(input_data, filters):
  
    x1 = tf.keras.layers.Conv2D(filters, 3, activation=tf.nn.relu, padding='same')(input_data)
    x1 = tf.nn.leaky_relu(x1, alpha=0.01, name='Leaky_ReLU') 
    x2 = tf.keras.layers.BatchNormalization()(x1)
    x2 = tf.keras.layers.Dropout(0.3)(x2)
    
    x3 = tf.keras.layers.Conv2D(filters, 5, activation=None, padding='same')(x2)
    x3 = tf.nn.leaky_relu(x3, alpha=0.01, name='Leaky_ReLU') 
    x4 = tf.keras.layers.BatchNormalization()(x3)
    x4 = tf.keras.layers.Dropout(0.3)(x4)
  
    x5 = tf.keras.layers.Conv2D(filters, 1, activation=None, padding='same')(input_data)
    x5 = tf.nn.leaky_relu(x5, alpha=0.01, name='Leaky_ReLU') 

    x = tf.keras.layers.Add()([x4 , x5 ])
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    return x

# %% [code]
def res_net_block_2(input_data, filters):
  
    x1 = tf.keras.layers.Conv2D(filters, 3, activation=tf.nn.relu, padding='same', activity_regularizer=tf.keras.regularizers.l1(0.001))(input_data)
    x1 = tf.nn.leaky_relu(x1, alpha=0.01, name='Leaky_ReLU') 
    x2 = tf.keras.layers.BatchNormalization()(x1)
    x2 = tf.keras.layers.Dropout(0.3)(x2)
    
    x3 = tf.keras.layers.Conv2D(filters, 5, activation=None, padding='same',activity_regularizer=tf.keras.regularizers.l1(0.001))(input_data)
    x3 = tf.nn.leaky_relu(x3, alpha=0.01, name='Leaky_ReLU') 
    x4 = tf.keras.layers.BatchNormalization()(x3)
    x4 = tf.keras.layers.Dropout(0.3)(x4)
  
    x5 = tf.keras.layers.Conv2D(filters, 1, activation=None, padding='same',activity_regularizer=tf.keras.regularizers.l1(0.001))(input_data)
    x5 = tf.nn.leaky_relu(x5, alpha=0.01, name='Leaky_ReLU') 

    x = tf.keras.layers.Add()([x2 , x4 , x5 ])
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    return x

# %% [code]
def resnet(inputsize,outputsize,depth,model_type):
    inputs = tf.keras.layers.Input(shape=(inputsize,inputsize,1))
    x = tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu,activity_regularizer=tf.keras.regularizers.l1(0.001))(inputs)
    x = tf.nn.leaky_relu(x, alpha=0.01, name='Leaky_ReLU') 
    x = tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu,activity_regularizer=tf.keras.regularizers.l1(0.001))(x)
    x = tf.nn.leaky_relu(x, alpha=0.01, name='Leaky_ReLU') 
    x = tf.keras.layers.MaxPooling2D(3)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    num_res_net_blocks = depth
    for i in range(num_res_net_blocks):
        x = res_net_block_2(x, 64)
    x = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu,activity_regularizer=tf.keras.regularizers.l1(0.001))(x)
    x = tf.nn.leaky_relu(x, alpha=0.01, name='Leaky_ReLU') 
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    if (model_type != "root"):
        x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(0.75)(x)
    output = tf.keras.layers.Dense(outputsize, activation=tf.nn.softmax)(x)
    model = tf.keras.models.Model(inputs, output)
    return model

ResNet = True 
CNN = False
IMG_SIZE = 96
model_root = resnet(IMG_SIZE, 168,10,"root")  # Input imagesize, outputtensor size, depth
model_vowel = resnet(IMG_SIZE, 11,10,"vowel")
model_consonant = resnet(IMG_SIZE, 7,10,"consonant")

# %% [code]
tf.keras.utils.plot_model(model_root, to_file='model1.png')
tf.keras.utils.plot_model(model_vowel, to_file='model2.png')
tf.keras.utils.plot_model(model_consonant, to_file='model3.png')

# %% [code]
EPOCHS = 30
INIT_LR = 1e-3
BS = 128
# initialize the optimizer and compile the model
print("[INFO] compiling models...")
opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model_root.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
model_vowel.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
model_consonant.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# %% [markdown]
# ## Read image data from feather format, binarize the labels, do train test split, resize the images, pass thru ImageDataGenerator and do model.fit_generator in a loop for the 2 sets of images

# %% [code]
histories = []

es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

for i in range(2):
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
        resized_img = tf.image.resize(trainX[j],[IMG_SIZE,IMG_SIZE])
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
            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    datagen.fit(resized_image)

    print("[INFO] creating validation dataset...")
    testX=np.array(testX).reshape(-1,137,236,1)
    print(testX.shape)
    resized_image_test=[]
    for i in range(len(testX)):
        resized_img = tf.image.resize(testX[i],[IMG_SIZE,IMG_SIZE])
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
model_root.save('model_root.h5')
model_vowel.save('model_vowel.h5')
model_consonant.save('model_consonant.h5')

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