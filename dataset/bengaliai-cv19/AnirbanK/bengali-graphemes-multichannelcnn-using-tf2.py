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
EPOCHS = 30
INIT_LR = 1e-3
BS = 256

es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
# %% [markdown]
# ## Simple model with multiple Outputs branching out from the Dense layers

# %% [code]
def build_model():
    inputs = tf.keras.layers.Input(shape = (96, 96, 1))

    x0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu)(inputs)
    x0 = tf.keras.layers.BatchNormalization(axis=-1)(x0)
    x0 = tf.keras.layers.MaxPooling2D((2, 2))(x0)
    print(x0.shape)
    x0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu)(x0)
    x0 = tf.keras.layers.BatchNormalization(axis=-1)(x0)
    x0 = tf.keras.layers.MaxPooling2D((2, 2))(x0)
    print(x0.shape)
    x0 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu)(x0)
    x0 = tf.keras.layers.BatchNormalization(axis=-1)(x0)
    x0 = tf.keras.layers.MaxPooling2D((2, 2))(x0)
    print(x0.shape)
    x0 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation=tf.nn.relu)(x0)
    x0 = tf.keras.layers.BatchNormalization(axis=-1)(x0)
    x0 = tf.keras.layers.MaxPooling2D((2, 2))(x0)
    print(x0.shape)
    x0 = tf.keras.layers.Dropout(rate=0.5)(x0)
    print(x0.shape)
    x0 = tf.keras.layers.Flatten()(x0)
    print(x0.shape)
    x = tf.keras.layers.Dense(1024, activation = tf.nn.relu)(x0)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation = tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation = tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    head_root = tf.keras.layers.Dense(168, activation = tf.nn.softmax,name="grapheme_output")(x)
    
    x1 = tf.keras.layers.Dense(1024, activation = tf.nn.relu)(x0)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dense(512, activation = tf.nn.relu)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dense(256, activation = tf.nn.relu)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dense(128, activation = tf.nn.relu)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dense(64, activation = tf.nn.relu)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dense(32, activation = tf.nn.relu)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dropout(rate=0.5)(x1)
    head_vowel = tf.keras.layers.Dense(11, activation = tf.nn.softmax,name="vowel_output")(x1)
    
    x2 = tf.keras.layers.Dense(1024, activation = tf.nn.relu)(x0)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dense(512, activation = tf.nn.relu)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dense(256, activation = tf.nn.relu)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dense(128, activation = tf.nn.relu)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dense(64, activation = tf.nn.relu)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dense(32, activation = tf.nn.relu)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dropout(rate=0.5)(x2)
    head_consonant = tf.keras.layers.Dense(7, activation = tf.nn.softmax,name="consonant_output")(x2)

    model = tf.keras.models.Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
    return model

# %% [code]
model = build_model()
# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
    "grapheme_output": "categorical_crossentropy",
    "vowel_output": "categorical_crossentropy",
    "consonant_output": "categorical_crossentropy"
}
lossWeights = {"grapheme_output": 1.0, "vowel_output": 1.0, "consonant_output":1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,metrics=["accuracy"])

tf.keras.utils.plot_model(model, to_file='model.png')

# %% [markdown]
# ## Read image data from feather format, binarize the labels, train on full data, create TF Dataset from images and labels, and do model.fit in a loop for the 4 sets of images

# %% [code]
histories = []

for i in range(4):
    print("iteration:"+str(i))

    graphemeLabels = []
    vowelLabels = []
    consonantLabels = []   
    train_df = pd.merge(read_data(i), train_df_, on='image_id').drop(['image_id','grapheme'], axis=1)
    train_df = train_df.astype('uint8')

# %% [code]
    graphemeLabels = train_df.grapheme_root
    vowelLabels = train_df.vowel_diacritic
    consonantLabels = train_df.consonant_diacritic
    # binarize all three sets of labels
    print("[INFO] binarizing labels...")
    graphemeLabels = graphemeLB.transform(np.array(graphemeLabels))
    vowelLabels = vowelLB.transform(np.array(vowelLabels))
    consonantLabels = consonantLB.transform(np.array(consonantLabels))

    print(graphemeLabels.shape)
    print(vowelLabels.shape)
    print(consonantLabels.shape)

    train_df=train_df.drop(["consonant_diacritic","grapheme_root","vowel_diacritic"],axis=1)

    trainX=np.array(train_df).reshape(-1,137,236,1)
    del train_df
    gc.collect()

    print("[INFO] creating train dataset...")
    print(trainX.shape)
    resized_image=[]
    for j in range(trainX.shape[0]):
        resized_img = tf.image.resize(trainX[j],[96,96])
        resized_img=np.array(resized_img)/255.
        resized_image.append(resized_img)

    train_set=tf.data.Dataset.from_tensor_slices(resized_image).batch(BS).prefetch(None)

    train_Y=tf.data.Dataset.from_tensor_slices((graphemeLabels, vowelLabels, consonantLabels)).batch(BS).prefetch(None)

    del trainX
    del resized_image
    del graphemeLabels
    del vowelLabels
    del consonantLabels
    gc.collect()

    dataset  = tf.data.Dataset.zip((train_set, train_Y))

    del train_set
    del train_Y
    gc.collect()

    print("[INFO] Model.fit starting...")

    history=model.fit(dataset,epochs=EPOCHS,verbose=2,callbacks=[es])

    histories.append(history)
    
# %% [code]
print("[INFO] Now saving Model")
model.save('Bengali_model_Tf2.h5')

# %% [code]
def plot_loss(his, epoch, title):
    plt.style.use('ggplot')
    fig=plt.figure()
    
    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')
    plt.plot(np.arange(0, epoch), his.history['grapheme_output_loss'], label='train_root_loss')
    plt.plot(np.arange(0, epoch), his.history['vowel_output_loss'], label='train_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['consonant_output_loss'], label='train_consonant_loss')

  
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    fig.savefig('plot_loss.png')

def plot_acc(his, epoch, title):
    plt.style.use('ggplot')
    fig=plt.figure()
    plt.plot(np.arange(0, epoch), his.history['grapheme_output_accuracy'], label='train_root_acc')
    plt.plot(np.arange(0, epoch), his.history['vowel_output_accuracy'], label='train_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['consonant_output_accuracy'], label='train_consonant_accuracy')
    
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()
    fig.savefig('plot_acc.png')

# %% [code]
for dataset in range(4):
    plot_loss(histories[dataset], EPOCHS, f'Training Dataset: {dataset}')
    plot_acc(histories[dataset], EPOCHS, f'Training Dataset: {dataset}')