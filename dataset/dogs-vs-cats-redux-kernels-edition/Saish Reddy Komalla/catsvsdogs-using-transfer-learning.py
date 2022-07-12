import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import os
import numpy as np
import cv2
import pandas as pd
import zipfile
from os import getcwd

train = f"{getcwd()}/../kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
# For extracting train images

local_train = train
z = zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", 'r')
z.extractall('/kaggle/working')
z.close()

# For extracting test images
local_train = train
z = zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", 'r')
z.extractall('/kaggle/working')
z.close()

#v 4.0
source = "/kaggle/working/train/"
os.mkdir("/kaggle/working/training")
os.mkdir("/kaggle/working/training/cats")
os.mkdir("/kaggle/working/training/dogs")
for i in os.listdir(source):
    if 'cat' in i:
        shutil.copyfile(source+ i , "/kaggle/working/training/" + "cats/"+ i)
    else:
        shutil.copyfile(source + i, "/kaggle/working/training/" + "dogs/" + i) 


# V 4.0 Transfer Learning
#Import ImageNet weights without the top fully connnected layer for us to train for our use
from tensorflow.keras.applications.inception_v3 import InceptionV3
pre_trained_model = InceptionV3(input_shape = (224,224,3), include_top = False, weights = "imagenet")

#freeze these layers as they dont need to be trained
for layer in pre_trained_model.layers:
  layer.trainable = False
pre_trained_model.summary()

#We take the convs and trian them ourselves
last_layer = pre_trained_model.get_layer('mixed7')
print(last_layer.output.shape)
last_output = last_layer.output

class mcb(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('acc') > 0.97:
      print('YOO')
      self.model.stop_training = True
callbacks = mcb()


#v 4.0
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dense(1, activation = 'sigmoid')(x)
model = Model(pre_trained_model.input, x)
model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr=0.001), metrics = ['acc'])
model.summary()

# V 4.0
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1.0/ 255.0)
train_generator = train_datagen.flow_from_directory(
    "/kaggle/working/training",
    batch_size = 32,
    class_mode = 'binary',
    target_size = (224,224)
)

#Steps should be the size of train_generator
model.fit_generator(train_generator, epochs = 15, steps_per_epoch=len(train_generator), callbacks = [callbacks])

#Save our trained model
model.save('fmodel_transfer_learning_t1.h5')
#Save weights if needed
# model.save_weights('fmodel_transfer_learning_weights.h5')

#  For prediction follow this link [here](https://stackoverflow.com/questions/52270177/how-to-use-predict-generator-on-new-images-keras)
#make test generator
os.mkdir("/kaggle/working/testing")
os.mkdir("/kaggle/working/testing/test")
for i in os.listdir("/kaggle/working/test"):
    shutil.copyfile("/kaggle/working/test/" + i, "/kaggle/working/testing/test/" + i)

test_datagen = ImageDataGenerator(rescale = 1.0/ 255.0)
test_generator = test_datagen.flow_from_directory(
    "/kaggle/working/testing",
    target_size = (224,224),
    batch_size = 10,
    class_mode = 'binary',
    shuffle = False
)
test_generator.reset()

#steps should be the size of test_generator
preds = model.predict_generator(test_generator, verbose  = 1, steps = len(test_generator))

c1 = np.round(preds).astype(int)

filenames = test_generator.filenames

results = pd.DataFrame({'file': filenames, 'pr': preds[:,0], 'cl': c1[:,0]})

c = range(1, 12500+ 1)

sol = pd.DataFrame({'id': c, "label": c1[:,0]})
sol.to_csv("CatsvsDogs_transfer_learning.csv", index=False)
