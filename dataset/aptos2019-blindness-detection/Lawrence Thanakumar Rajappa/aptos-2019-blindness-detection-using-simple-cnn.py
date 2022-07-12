# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
from zipfile import ZipFile
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


print(os.listdir("../input"))
dataset = []
labels = []
def prepare_Images(label,path):
    img=cv2.imread(path,cv2.IMREAD_COLOR)
    img_res=cv2.resize(img,(50,50))
    img_array = img_to_array(img_res)
    img_array = img_array/255
    dataset.append(img_array)
    labels.append(str(label))

train_Data = pd.read_csv("../input/train.csv")
train_Data.head()

id_code_Data = train_Data['id_code']
diagnosis_Data = train_Data['diagnosis']

for id_code,diagnosis in tqdm(zip(id_code_Data,diagnosis_Data)):
    path = os.path.join('../input/train_images','{}.png'.format(id_code))
    prepare_Images(diagnosis,path)

#Convert list to numpy array
images = np.array(dataset)
label_arr = np.array(labels)

#spliting the training data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(images,label_arr,test_size=0.20,random_state=42)

#Convert to class labels categorical
y_train = np_utils.to_categorical(y_train, num_classes=5)
y_test = np_utils.to_categorical(y_test, num_classes=5)

#Building model
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(5,activation="softmax"))#5 represent output layer neurons
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train,y_train,batch_size=64,epochs=10,verbose=1,validation_data=(x_test, y_test))

pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
score = round(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)),2)
print(score)
report = classification_report(y_test.argmax(axis=1), pred.argmax(axis=1))
print(report)
conMat = confusion_matrix(y_test.argmax(axis=1),pred.argmax(axis=1))
print(conMat)

# Any results you write to the current directory are saved as output.
test_df = pd.read_csv('../input/test.csv')
test_df.head()

x = test_df['id_code']
test_Dataset = []
def make_test_data(path):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    img_res = cv2.resize(img, (50,50))
    img_array = img_to_array(img_res)
    img_array = img_array/255
    test_Dataset.append(img_array)

for id_code in tqdm(x):
    path = os.path.join('../input/test_images','{}.png'.format(id_code))
    make_test_data(path)
test_image = np.array(test_Dataset)
pred=model.predict(test_image)
pred=np.argmax(pred,axis=1)
pred

sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df.head()

sub_df.diagnosis = pred
sub_df.head()

sub_df.to_csv("submission.csv",index=False)