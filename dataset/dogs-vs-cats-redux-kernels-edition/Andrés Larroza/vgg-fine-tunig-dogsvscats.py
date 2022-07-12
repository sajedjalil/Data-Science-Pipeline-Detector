# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import shutil
import math

original_train_dir = '../input/dogs-vs-cats-redux-kernels-edition/train/train/'
train_cats = ['{}'.format(i) for i in os.listdir(original_train_dir) if 'cat' in i] 
train_dogs = ['{}'.format(i) for i in os.listdir(original_train_dir) if 'dog' in i] 

train_dir = 'trainDogsCats'
os.mkdir(train_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

for file in train_cats:
    orig = os.path.join(original_train_dir, file)
    dest = os.path.join(train_cats_dir, file)
    shutil.copyfile(orig, dest)
    
for file in train_dogs:
    orig = os.path.join(original_train_dir, file)
    dest = os.path.join(train_dogs_dir, file)
    shutil.copyfile(orig, dest)

import time
from PIL import Image
from keras import applications
from keras.preprocessing.image import ImageDataGenerator, image
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense


# dimensions of our images.
img_width, img_height = 150, 150
epochs = 50
batch_size = 25
nb_train_samples = len(train_cats)+len(train_dogs)
#####################
# def prepare_images(list_of_images):
#     x = [] # resized images
#     y = [] # labels
    
#     for im in list_of_images:
#         x.append(image.img_to_array(image.load_img(im, target_size=(img_width,img_height))))
#         if 'dog' in im:
#             y.append(1)
#         elif 'cat' in im:
#             y.append(0)
      
#     return x, y

# train_dir = '../input/dogs-vs-cats-redux-kernels-edition/train/train/'
# train_images = [train_dir + '{}'.format(i) for i in os.listdir(train_dir)] #training images
# x_train,y_train = prepare_images(train_images)


#Calculate bottleneck features
datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
model = applications.VGG16(weights=None, include_top=False,  input_shape = (img_width,img_height,3))
model.load_weights('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

generator = datagen.flow_from_directory(train_dir,target_size=(img_height, img_width),batch_size=batch_size,shuffle=False)
bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
    #np.save(open('../input/bottleneck_features_train.npy', 'wb'),bottleneck_features_train)

#train top_model
    #train_data = np.load(open('../input/bottleneck_features_train.npy','rb'))
train_data = np.array(bottleneck_features_train)
train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
    
    
model.fit(train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,verbose=2)

top_model_weights_path = '../input/bottleneck_fc_model.h5'
model.save_weights(top_model_weights_path)


# build the VGG16 network
base_model = applications.VGG16(weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False,  input_shape = (img_width,img_height,3))

for layer in base_model.layers[:15]:
    layer.trainable = False
for layer in base_model.layers:
    print(layer, layer.trainable)

# Create the model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
top_model.set_weights(model.get_weights())

# # Combine base and top
model = Sequential()
model.add(base_model)
model.add(top_model)


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
start_time = time.time()
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,verbose=2)
print("--- %s seconds ---" % (time.time() - start_time))

fineTune_model_weights_path = 'fineTune_model.h5'
model.save_weights(fineTune_model_weights_path)

#Predict test
test_dir = '../input/dogs-vs-cats-redux-kernels-edition/test'
test_datagen = image.ImageDataGenerator(rescale=1. / 255)

i = 0
ids = []
test_prob = []
gen = test_datagen.flow_from_directory(test_dir, target_size=(img_height, img_width), batch_size=batch_size, shuffle=False, class_mode=None)
submission = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

for batch in gen:
    pred = model.predict_classes(batch)
    prob = model.predict(batch)
    test_prob.extend(prob)
    
    i += 1
    if i == math.ceil(len(submission['id']) / batch_size):
        break

test_probs = [item[0] for item in test_prob]

submission = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
submission['id'] = ['{}'.format(s[s.find('/')+1:s.rfind('.')]) for s in gen.filenames] 
submission['label'] = test_probs
submission.to_csv("submission.csv", index=False)

if os.path.isdir(train_dir):
    shutil.rmtree(train_dir)