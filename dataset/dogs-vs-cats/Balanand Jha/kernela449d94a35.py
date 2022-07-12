import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
import cv2
from tqdm import tqdm
import h5py
path="../input/train/train"
new_path_train="../input/train/train_new_resized_96"
label=[]
data1=[]
counter=0
for file in os.listdir(path):
    image_data=cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE)
    image_data=cv2.resize(image_data,(96,96))
    if file.startswith("cat"):
        label.append(0)
    elif file.startswith("dog"):
        label.append(1)
    try:
        data1.append(image_data/255)
    except:
        label=label[:len(label)-1]
    counter+=1
    if counter%1000==0:
        print(counter,"Image data retrived")
        
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
model = Sequential()
model.add(Conv2D(kernel_size=(3,3),input_shape=(96,96,1),filters=3,activation='relu'))
model.add(Conv2D(kernel_size=(3,3),filters=10,activation ='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(kernel_size=(3,3),filters=3,activation="relu"))
model.add(Conv2D(kernel_size=(5,5),filters=5,activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(kernel_size=(2,2),strides=(2,2),filters=10))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.summary()

model.compile(optimizer="adadelta",loss="binary_crossentropy",metrics=["accuracy"])

import numpy as np
data1=np.array(data1)
print(data1.shape)
data1=data1.reshape((data1.shape)[0],(data1.shape)[1],(data1.shape)[2],1)
labels=np.array(label)

history = model.fit(data1,labels,validation_split=0.25,epochs=10,batch_size=10)
history.history.keys()
model.save_weights("model.h5")

test_data=[]
id=[]
counter=0
for file in os.listdir("../input/test1/test1"):
    image_data=cv2.imread(os.path.join("../input/test1/test1",file), cv2.IMREAD_GRAYSCALE)
    try:
        image_data=cv2.resize(image_data,(96,96))
        test_data.append(image_data/255)
        id.append((file.split("."))[0])
    except:
        print ("one down")
    counter+=1
    if counter%1000==0:
        print (counter," image data retreived")
test_data1=np.array(test_data)
print (test_data1.shape)
test_data1=test_data1.reshape((test_data1.shape)[0],(test_data1.shape)[1],(test_data1.shape)[2],1)

dataframe_output=pd.DataFrame({"id":id})

predicted_labels=model.predict(test_data1)

predicted_labels=np.round(predicted_labels,decimals=2)

labels=[1 if value>0.5 else 0 for value in predicted_labels]

dataframe_output["label"]=labels


dataframe_output.to_csv("submission.csv",index=False)
 
#plotting graph   
import matplotlib.pyplot as plt 

#summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()















