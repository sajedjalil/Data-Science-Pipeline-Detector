# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import itertools


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

img_rows, img_cols = 28, 28

img_size = 224

seq_model = Sequential()

seq_model.add(ResNet50(include_top = False, pooling = 'avg', weights = resnet_weights_path))

#seq_model.add(Conv2D(12, kernel_size = 3, activation = 'relu'))

seq_model.add(Dense(num_classes, activation = 'softmax'))

seq_model.layers[0].trainable = False

seq_model.compile(optimizer = 'sgd',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])


data_generator = ImageDataGenerator(preprocessing_function = preprocess_input, validation_split = 0.2)

train_generator = data_generator.flow_from_directory(
                                    directory = '../input/dogs-vs-cats-redux-kernels-edition/train',
                                    target_size = (img_size, img_size),
                                    batch_size = 100,
                                    class_mode = 'categorical',
                                    #shuffle = True,
                                    subset = 'training')

val_generator = data_generator.flow_from_directory(
                                    directory = '../input/dogs-vs-cats-redux-kernels-edition/train',
                                    target_size = (img_size, img_size),
                                    class_mode = 'categorical',
                                    #shuffle = True,
                                    subset = 'validation')

print('Fit Stats')
                                    
fit_stats = seq_model.fit_generator(train_generator,
                                    steps_per_epoch = 10,
                                    validation_data = val_generator,
                                    validation_steps = 1)

print('Train Generator')
                                    
test_generator = data_generator.flow_from_directory(
                                    directory = '../input/dogs-vs-cats-redux-kernels-edition/test',
                                    target_size = (img_size, img_size),
                                    class_mode = 'categorical')

print('Predictions')
                                    
pred = seq_model.predict_proba(test_generator,
                                 batch_size = 32,
                                 verbose = 1)

print(len(test_generator))
print(pred)

index = [x for x in range(1, 12501)]

output = pd.DataFrame({'id': index,
                       'label': pred[:, 1]})
                       
output.to_csv('submission.csv', index=False)
print(output.shape)