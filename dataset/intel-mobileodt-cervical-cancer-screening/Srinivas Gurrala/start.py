#import cv2 as cv
from matplotlib import pyplot as pp
from matplotlib import colors as pc
import numpy as np
import scipy.misc as scmisc
import os
import pandas as pd


from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 

 

 
# dimensions of our images. 
img_width, img_height = 200, 200 

 
train_data_dir = '../input/train' 

nb_train_samples = 200
nb_validation_samples = 25
epochs = 3
batch_size = 50

 
#if K.image_data_format() == 'channels_first': 
#    input_shape = (3, img_width, img_height) 
#else: 
#    input_shape = (img_width, img_height, 3) 
input_shape = (img_width,img_height,3)
def preprocessImage(ii):
    #ii=scmisc.imread(img).astype(np.uint8)
    ii = scmisc.imresize(ii,(200,200),interp='bilinear')
    ii1 = pc.rgb_to_hsv(ii)
    indices = np.where(ii1[:,:,0]<0.5)
    ii[:,:,0][indices]=0
    ii[:,:,1][indices]=0
    ii[:,:,2][indices]=0
    return(ii.astype(np.uint8))
    
model = Sequential() 
model.add(Conv2D(32,8,2))#,input_shape=input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(4,4))) 

model.add(Conv2D(16, 8, 2)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
#
#
#model.add(Conv2D(64, (16, 16))) 
#model.add(Activation('relu')) 
#model.add(MaxPooling2D(pool_size=(2, 2))) 

 
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(3)) 
model.add(Activation('softmax')) 

 
model.compile(loss='mean_squared_error', optimizer='sgd')

 
# this is the augmentation configuration we will use for training 
train_datagen = ImageDataGenerator(preprocessing_function = preprocessImage,
    data_format='channels_last',
    horizontal_flip=False) 

 
# this is the augmentation configuration we will use for testing: 
# only rescaling 
test_datagen = ImageDataGenerator(
    preprocessing_function = preprocessImage,
    data_format='channels_last',
    horizontal_flip=False)

train_generator = train_datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
     class_mode='categorical') 
 
 
model.fit_generator( 
     train_generator, 
     steps_per_epoch=nb_train_samples // batch_size, 
     epochs=epochs)  
        

path = '../input/test' 
images = os.listdir(path)
matching = [s for s in images if "jpg" in s]
pred = np.array([])
ind = 0
for ii in matching:
    i1 = scmisc.imread(path+'/'+ii)
    i2 = preprocessImage(i1)
    if ind == 0:
        pred = model.predict_proba(i2.reshape((1,img_width, img_height,3)))
        ind=ind+1
    else:
        pred1 = model.predict_proba(i2.reshape((1,img_width, img_height,3)))
        pred = np.concatenate((pred,pred1))
    

df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
df['image_name'] = matching
df.to_csv('submission.csv', index=False)
print("submission created")
