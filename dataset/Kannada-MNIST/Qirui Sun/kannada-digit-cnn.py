# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from time import process_time_ns 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_set = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
val_set = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

train_label = train_set.label
train_data = train_set.drop('label',axis =1)

val_label = val_set.label
val_data = val_set.drop("label",axis = 1)

test = test.drop('id',axis = 1)



train_data /= 255.
val_data /= 255.
test /= 255.

train_data = train_data.values.reshape(-1,28,28,1)
val_data = val_data.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

train_img,test_img,train_id,test_id = train_test_split(train_data,train_label,
                                                       test_size=0.2,random_state=666)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

t1_start = process_time_ns()
model.fit(train_img,train_id,epochs=8)
t1_end = process_time_ns()
print("fit time is %f"%((t1_end-t1_start)/1000000000))

model.save('/kaggle/working/my_model.h5')
# model_0 =keras.models.load_model('saved_model/my_model.h5')


test_loss, test_acc = model.evaluate(test_img,  test_id, verbose=2)

t2_start = process_time_ns()
pred = model.predict(test)
t2_end = process_time_ns()
print("predict time is %f"%((t2_end - t2_start)/1000000000))
# test_result = np.argmax(pred,axis=1)

test_result = np.argmax(pred,axis=1)

pd_outcome = pd.DataFrame({'id':[i for i in range(len(test_result))],'label':test_result})
pd_outcome.to_csv('/kaggle/working/submission.csv',index = False)