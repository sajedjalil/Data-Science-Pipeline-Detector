import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import gzip
import pandas as pd
from tensorflow.keras.layers import Input,Dense,Conv2D, MaxPooling2D, UpSampling2D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model,Sequential
import tensorflow as tf

plt.style.use('ggplot')
def imshow(img):
    plt.imshow(img, cmap='gray')
img = cv2.imread('../input/train/cat.1.jpg', 0)
img = cv2.resize(img, (150, 150))
imshow(img)

sh=[]
img = cv2.imread('../input/train/cat.1.jpg')
img = cv2.resize(img, (150, 150))
sh.append(img)
img = cv2.imread('../input/train/cat.1000.jpg')
img = cv2.resize(img, (150, 150))
sh.append(img)
img=np.array(sh)
print(img.shape)
def CreateModel():
  model=Sequential()
  model.add(Conv2D(32, (2,2), input_shape=(150,150,3),activation="relu",padding="same"))
  model.add(MaxPooling2D((3,3),padding="same"))
  model.add(Conv2D(64, (5,5),activation="relu",padding="same"))
  model.add(MaxPooling2D((3,3),padding="same"))
  model.add(Conv2D(128, (5,5),activation="relu",padding="same"))
  model.add(MaxPooling2D((5,5),padding="same"))
  model.add(Flatten())
  model.add(Dense(1000,activation="relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))
  model.add(Dense(500,activation="relu"))
  model.add(BatchNormalization())
  model.add(Dense(250,activation="relu"))
  model.add(Dense(125,activation="relu"))
  model.add(Dense(75,activation="relu"))
  model.add(Dense(1,activation="sigmoid"))
  model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
  return model

model=CreateModel()
model.summary()

def generate_data():
    test_imgs = []
    train_imgs = []
    train_label = []
    files = os.listdir('../input/train/')
    for filename in files:
        img = cv2.imread('../input/train/'+filename)
        img = cv2.resize(img, (150, 150))
        train_imgs.append(img)
        if "dog" in filename:
          train_label.append(1)
        elif "cat" in filename:
          train_label.append(0)
       
    train_imgs = np.array(train_imgs)
    train_label=np.array(train_label)
    
    files = os.listdir('../input/test/')
    for filename in files:
        img = cv2.imread('../input/test/'+filename)
        img = cv2.resize(img, (150, 150))
        test_imgs.append(img)

    test_imgs = np.array(test_imgs)
    
    return train_imgs, train_label, test_imgs
  
train ,label ,test =generate_data()
print(train.shape)
train = train.astype("float32")/255.0
test = test.astype("float32")/255.0

model.fit(train,label,epochs=20,batch_size=500,verbose=1,validation_split=0.2)
import gc
gc.collect()
result=model.predict(test)
rs=[]
SI=[]
name=[]
print(result.shape)
print(result.shape)
for i in range(0,result.shape[0]):
  rs.append(round(result[i][0]))
  if round(result[i][0]) == 1:
    name.append("dog")
  else:
    name.append("cat")
  SI.append(i)

output = pd.DataFrame( data={"Id":SI, "label":rs,"label_name":name} )
print(output)
