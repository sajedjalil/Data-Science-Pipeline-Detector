import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import os
from shutil import copyfile,copy,copy2
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Model
model = ResNet50(weights='imagenet',include_top=False, input_shape=(224, 224, 3))

x = model.output
x = Flatten()(x)
x=Dropout(0.25)(x)
x=Dense(units=1000,activation='relu')(x)
x=Dropout(0.25)(x)
x=Dense(units=750,activation='relu')(x)
x=Dropout(0.25)(x)
x=Dense(units=750,activation='relu')(x)
x=Dropout(0.25)(x)
x=Dense(units=750,activation='relu')(x)
x=Dropout(0.25)(x)
x=Dense(units=1000,activation='relu')(x)
x=Dropout(0.6)(x)
#clf.add(Dense(units=120,activation='softmax')
#stochastic gradient descent -Adam -optimizer
#loss func categorical cross entropy
#metrics = accuracy
#clf.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
predictions = Dense(120, activation='softmax')(x)

import keras
main_model = Model(inputs=model.input, outputs=predictions)
#train only the hidden layers and output layer, donot train the resnet model
for curLayer in model.layers:
    curLayer.trainable = False

main_model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
main_model.summary()


labels = pd.read_csv('../input/labels.csv')

labels_dict = {i:j for i,j in zip(labels['id'],labels['breed'])}
classes = set(labels_dict.values())
images = [f for f in os.listdir('../input/train')]
#(images)
#os.makedirs('training_images')
#os.makedirs('validation_images')

if not os.path.exists('training_images'):
    os.makedirs('training_images')

if  not os.path.exists('validation_images'):
    os.makedirs('validation_images')
from os.path import join


for item in images:
    filekey = os.path.splitext(item)[0]
    if not os.path.exists('training_images/'+labels_dict[filekey]):
        os.makedirs('training_images/'+labels_dict[filekey])
    if not os.path.exists('validation_images/'+labels_dict[filekey]):
        os.makedirs('validation_images/'+labels_dict[filekey])

count = 0 
destination_directory = 'training_images'
cwd = os.getcwd()
print(cwd)
for item in images:
    if count >7999:
        destination_directory = 'validation_images'
    filekey = os.path.splitext(item)[0]
    dest_file_path = join(destination_directory,labels_dict[filekey],item)
    src_file_path = join("..","input","train",item)
    #print(src_file_path)
    if not os.path.exists(dest_file_path):
        copyfile(src_file_path, dest_file_path)
    count +=1
    

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_images',
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'validation_images',
        target_size=(224, 224),
        batch_size=22,
        class_mode='categorical')
        

hist=main_model.fit_generator(
        training_set,
        steps_per_epoch=400,
        epochs=1,
        validation_data=test_set,
        validation_steps=101)
#callbacks=[early_stopping_monitor])
import shutil
dir_name1 = "training_images"
dir_name2 = "validation_images"
if os.path.isdir(dir_name1):
    shutil.rmtree(dir_name1)
    
    
if os.path.isdir(dir_name2):
    shutil.rmtree(dir_name2)
import cv2
test_set_list = []
test_set_ids = []
for curImage in os.listdir('../input/test'):
    test_set_ids.append(os.path.splitext(curImage)[0])
    print(os.path.splitext(curImage)[0])
    curImage = cv2.imread('../input/test/'+curImage)
    test_set_list.append(cv2.resize(curImage,(224, 224)))
    
test_set_list = np.array(test_set_list, np.float32)/255.0
predictions= main_model.predict(test_set_list)

classes= {index:breed for breed,index in training_set.class_indices.items()}
column_names = [classes[i] for i in range(120)]
predictions_df = pd.DataFrame(predictions)
predictions_df.columns = column_names
predictions_df.insert(0,'id', test_set_ids)
predictions_df.to_csv('resnet_5_submission.csv',sep=",")
print("all done")


