# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
print(data.shape)

test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
print(test_data.shape)

train = data[:]
val = data[55000:]
train_label = np.float32(train.label)
val_label = np.float32(val.label)
train_image = np.float32(train[train.columns[1:]])
val_image = np.float32(val[val.columns[1:]])
test_image = np.float32(test_data[test_data.columns[1:]])
print('train shape: %s'%str(train.shape))
print('val shape: %s'%str(val.shape))
print('train_label shape: %s'%str(train_label.shape))
print('val_label shape: %s'%str(val_label.shape))
print('train_image shape: %s'%str(train_image.shape))
print('val_image shape: %s'%str(val_image.shape))
print('test_image shape: %s'%str(test_image.shape))

train_image = train_image/255.0
val_image = val_image/255.0
test_image = test_image/255.0

train_image = train_image.reshape(train_image.shape[0],28,28,1)
val_image = val_image.reshape(val_image.shape[0],28,28,1)
test_image = test_image.reshape(test_image.shape[0],28,28,1)

print('train_image shape: %s'%str(train_image.shape))
print('train_image shape: %s'%str(train_image.shape))
print('val_image shape: %s'%str(val_image.shape))

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
          
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(10, activation = "softmax"))

model.summary()


from keras.utils import to_categorical

train_image = train_image.reshape((60000, 28, 28, 1))
train_image = train_image.astype("float32") / 255

test_image = test_image.reshape((5000, 28, 28, 1))
test_image = val_image.astype("float32") / 255

train_labels = to_categorical(train_label)
test_labels = to_categorical(val_label)

model.compile(optimizer = "rmsprop",
             loss = "categorical_crossentropy",
             metrics = ["accuracy"])

model.fit(train_image, train_labels, epochs = 5, batch_size = 64)

test_loss, test_acc = model.evaluate(test_image, test_labels)
test_acc

sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
raw_test_id=test_data.id
test_data=test_data.drop("id",axis="columns")
test_data=test_data / 255
test=test_data.values.reshape(-1,28,28,1)
test.shape

sub=model.predict(test)     ##making prediction
sub=np.argmax(sub,axis=1) ##changing the prediction intro labels

sample_sub['label']=sub
sample_sub.to_csv('submission.csv',index=False)

