# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time, gc
import tensorflow as tf
from PIL import Image
from tqdm.auto import tqdm
from glob import glob
import cv2
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
# ## Exploratory Data Analysis

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

es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
# %% [code]
EPOCHS = 30
INIT_LR = 1e-3
BS = 256


# detect and init the TPU
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#tf.config.experimental_connect_to_cluster(tpu)
#tf.tpu.experimental.initialize_tpu_system(tpu)
    
# instantiate a distribution strategy
#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

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
#with tpu_strategy.scope():
model = build_model()
tf.keras.utils.plot_model(model, to_file='model.png')
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

# %% [code]
class MultiOutputDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):

    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):

        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)


        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,
                                         shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict

# %% [markdown]
# ## Read image data from feather format, use ImageDataGenerator instance to create TF Dataset using from_generator, and do model.fit in a loop for the 4 sets of images

# %% [code]
IMG_SIZE = 96
N_CHANNELS = 1

histories = []

for i in range(2):
    print("iteration:"+str(i))
    graphemeLabels = []
    vowelLabels = []
    consonantLabels = []
    X_train = pd.merge(read_data(i), train_df_, on='image_id').drop(['image_id','grapheme'], axis=1)
    X_train=X_train.astype('uint8')
    graphemeLabels = X_train.grapheme_root
    vowelLabels = X_train.vowel_diacritic
    consonantLabels = X_train.consonant_diacritic
    X_train=X_train.drop(["consonant_diacritic","grapheme_root","vowel_diacritic"],axis=1)

    # binarize all three sets of labels
    print("[INFO] binarizing labels...")
    graphemeLabels = graphemeLB.transform(np.array(graphemeLabels))
    vowelLabels = vowelLB.transform(np.array(vowelLabels))
    consonantLabels = consonantLB.transform(np.array(consonantLabels))

    print(graphemeLabels.shape)
    print(vowelLabels.shape)
    print(consonantLabels.shape)
    (trainX, testX, trainGraphemeY, testGraphemeY,trainVowelY, testVowelY,trainConsonantY,testConsonantY) = train_test_split(X_train, graphemeLabels, vowelLabels,consonantLabels,test_size=0.1, random_state=42)

    del graphemeLabels
    del vowelLabels
    del consonantLabels
    gc.collect()
    
    trainX=np.array(trainX).reshape(-1,137,236,1)
    print("[INFO] resizing train dataset...")
    trainX=np.array(trainX).reshape(-1,137,236,1)
    print(trainX.shape)
    resized_image=[]
    for j in range(len(trainX)):
        resized_img = tf.image.resize(trainX[j],[96,96])
        resized_img=np.array(resized_img)/255.
        resized_image.append(resized_img)
    resized_image = np.asarray(resized_image)
    
    del trainX
    gc.collect()

    testX=np.array(testX).reshape(-1,137,236,1)
    print("[INFO] creating validation dataset...")
    print(testX.shape)
    resized_image_test=[]
    for i in range(len(testX)):
        resized_img = tf.image.resize(testX[i],[96,96])
        resized_img=np.array(resized_img)/255.
        resized_image_test.append(resized_img)
    resized_image_test = np.asarray(resized_image_test)
    test_set=tf.data.Dataset.from_tensor_slices(resized_image_test).batch(BS).prefetch(None)
    test_Y=tf.data.Dataset.from_tensor_slices((testGraphemeY, testVowelY, testConsonantY)).batch(BS).prefetch(None)
    val_dataset  = tf.data.Dataset.zip((test_set, test_Y))
    
    del resized_image_test
    del test_set
    del testGraphemeY
    del testVowelY
    del testConsonantY
    del test_Y
    gc.collect()
    
    datagen = MultiOutputDataGenerator(
        rotation_range= 20,  
        width_shift_range=[-5,+5])  

    print("[INFO] Datagen on resized images...") 
    datagen.fit(resized_image)
   
    print("[INFO] Model.fit starting...")
    history=model.fit_generator(datagen.flow(resized_image,{'grapheme_output': trainGraphemeY, 'vowel_output': trainVowelY, 'consonant_output': trainConsonantY},batch_size=BS),
                   steps_per_epoch=resized_image.shape[0] // BS, epochs=EPOCHS, verbose=2, callbacks=[es], validation_data=val_dataset)
    

    del resized_image
    gc.collect()
    histories.append(history)
    print("iteration completed:"+str(i))

gc.collect()

# %% [code]
print("[INFO] Now saving Model")
model.save('Bengali_model_AugDSn.h5')

# %% [code]
def plot_loss(his, epoch, title):
    plt.style.use('ggplot')
    fig=plt.figure()
    
    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')
    plt.plot(np.arange(0, epoch), his.history['grapheme_output_loss'], label='train_root_loss')
    plt.plot(np.arange(0, epoch), his.history['vowel_output_loss'], label='train_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['consonant_output_loss'], label='train_consonant_loss')
    
    plt.plot(np.arange(0, epoch), his.history['val_grapheme_output_loss'], label='val_root_loss')
    plt.plot(np.arange(0, epoch), his.history['val_vowel_output_loss'], label='val_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['val_consonant_output_loss'], label='val_consonant_loss')
  
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
   
    plt.plot(np.arange(0, epoch), his.history['val_grapheme_output_accuracy'], label='val_root_acc')
    plt.plot(np.arange(0, epoch), his.history['val_vowel_output_accuracy'], label='val_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['val_consonant_output_accuracy'], label='val_consonant_accuracy')
    
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()
    fig.savefig('plot_acc.png')

# %% [code]
for dataset in range(2):
    plot_loss(histories[dataset], EPOCHS, f'Training Dataset: {dataset}')
    plot_acc(histories[dataset], EPOCHS, f'Training Dataset: {dataset}')