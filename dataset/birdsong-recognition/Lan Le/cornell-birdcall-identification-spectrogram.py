# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

"""
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import librosa
import librosa.display
import glob
import os
import matplotlib.pyplot as plt
from matplotlib import figure
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import model_from_json



!mkdir /kaggle/working/train

# convert audio files into pictures (spectrogram)
def create_spectrogram(file_name,name):
    try:
        plt.interactive(False)
        clip, sample_rate = librosa.load(file_name, sr=None)
        fig = plt.figure(figsize=[0.72,0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        file_name  = '/kaggle/working/train/' + name + '.jpg'
        plt.savefig(file_name, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close()    
        fig.clf()
        plt.close(fig)
        plt.close('all')
        del file_name, name, clip, sample_rate, fig, ax, S
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)


# generate image data from audio data
data_dir=np.array(glob.glob("../input/birdsong-recognition/train_audio/*/*.mp3"))


for file in tqdm(data_dir):
    filename, name = file, file.split('/')[-1].split('.')[0]
    create_spectrogram(filename, name)



# initialize image generator
def append_ext(fn):
    return fn.split('.')[0] + ".jpg"

traindf = pd.read_csv('../input/birdsong-recognition/train.csv',dtype = str)
traindf["filename"] = traindf["filename"].apply(append_ext)

datagen = ImageDataGenerator(rescale = 1./255., validation_split = 0.25)

train_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="/kaggle/working/train/",
    x_col="filename",
    y_col="species",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

num_class = len(set(train_generator.classes))


valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="/kaggle/working/train/",
    x_col="filename",
    y_col="species",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))


# build a CNN model for classification
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class, activation='softmax'))

adam = Adam(lr=0.0005, decay=1e-6)
model.compile(optimizer=adam, loss="categorical_crossentropy" ,metrics=["accuracy"])
model.summary()

# train the model
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=150)
model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")



# Reference: https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4