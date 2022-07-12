# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#==============================================================================
# System specs:
# Core I7-4720HQ 2.60GHz (3.60GHz with Turbo Boost)
# 32GB RAM
# GTX980M 4GB RAM
# 
# 1 Round running Time: Around 3 Hours
# 
#==============================================================================

#-*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:28:19 2017

@author: Denes
"""
# Import
from scipy import misc
from scipy import ndimage

import gc
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.layers.noise import GaussianNoise

#Set seed
np.random.seed(117)

#Functions
def readin_y(df, path):
    y = []
    print ("Reading in the Categories: ")
    for i in range(len(df)):
        if (i % 100 == 0) : print (i)
        y.append(df.ix[i][1])
    y = np.array(y)
    return (y)

def readin_x(df, path):
    x = []
    print ("Reading in the Training/Testing Data: ")
    for i in range(len(df)):
        if (i % 100 == 0) : print (i)
        img = misc.imread(path + str(int(df.ix[i][0])) + ".jpg")            
        x.append(img)    
    x = np.array(x)    
    return (x)

def ext_readin_x(df, path):
    x = []
    for i in np.arange(1, (len(df) / 2) + 1):
        if (i % 100 == 0) : print (i)
        img = misc.imread(path + str(int(df.ix[i][0])) + "a.jpg")        
        x.append(img)
        img = misc.imread(path + str(int(df.ix[i][0])) + "b.jpg")        
        x.append(img)
    x = np.array(x)    
    return (x)

def img_outlier_rotate(path, image, angle):
    img_outlier = []
    img_outlier = misc.imread(path + image + ".jpg")
    img_outlier = ndimage.interpolation.rotate(img_outlier, angle)

    plt.imshow(img_outlier)
    plt.show()
    return (img_outlier)
    
def img_crop_resize(df, path, folder):
    print ("Pre Processing the Data: ")    
    for i in range(len(df)):
        if (i % 100 == 0) : print (i)
        img = misc.imread(path + str(int(df.ix[i][0])) + ".jpg")

        crop_img_left = misc.imresize(img[0:866, 0:866], (224, 224, 3))
        misc.imsave(path + folder + str(i+1) + "a.jpg", crop_img_left)

        crop_img_right = misc.imresize(img[0:866, 288:1154], (224, 224, 3))
        misc.imsave(path + folder + str(i+1) + "b.jpg", crop_img_right)
        
def img_extend_y(train_y):
    j = 0
    m_y = np.zeros((len(train_y) * 2, 1), dtype = np.uint)
    for i in range(len(train_y)):
        m_y[j:(j + 2), 0] = train_y[i]
        j += 2
    return pd.DataFrame(m_y).astype(int)
        
def bulk_predict(test_xy):
    weight_list = glob.glob("F:/Code/Python/3 Image ISM/Models/Final/*.hdf5")
    end_test_p = np.zeros((len(test_xy), 0))
    for i in range(len(weight_list)):     
        print ("Weight list number: %d" % (i+1))
        model.load_weights(weight_list[i])
        v_pred = model.predict(test_xy, 
                               batch_size = 16, 
                               verbose = 1
                               )
        print ("\n")
        end_test_p = np.concatenate((end_test_p, v_pred), axis = 1)
    return(end_test_p)

def pred_vote(test_pred):
    m_y = np.zeros((int(len(test_pred)/2), 1), dtype = np.uint)
    v_preds = np.mean(test_pred, axis = 1)
    j = 0
    for i in range(len(m_y)):
        m_y[i, 0] = np.round(np.mean(v_preds[j:(j+2)]))
        j += 2
    return(m_y)

def write_to_file(pred_array):
    m_temp = np.zeros((len(pred_array), 2))
    
    for i in range(len(m_temp)):
        m_temp[i, 0] = i + 1
        m_temp[i, 1] = end_test_r[i]
        
    df_sample = pd.DataFrame(m_temp,columns = ['name', 'invasive']).astype(int)
    
    del m_temp, i
    
    df_sample.to_csv("F:/Code/Python/3 Image ISM/submission.csv", index = False)
    return print("Submission saved to csv file.")

#==============================================================================
# #Default Read-in and Pre processing
# labels = pd.read_csv("F:/Code/Python/3 Image ISM/train_labels.csv")
# train_path = ("F:/Code/Python/3 Image ISM/train/")
# 
# sample = pd.read_csv("F:/Code/Python/3 Image ISM/sample_submission.csv")
# test_path = ("F:/Code/Python/3 Image ISM/test/")
# 
# #img_o = img_outlier_rotate("F:/Code/Python/3 Image ISM/test/", "1068", 90)
# #misc.imsave("F:/Code/Python/3 Image ISM/test/1068.jpg", img_o)
# 
# train_x = readin_x(labels, train_path)
# train_y = readin_y(labels, train_path)
# 
# test_x = readin_x(sample, test_path)
# test_y = readin_y(sample, test_path)
# 
# img_crop_resize(labels, train_path, "pp/")
# img_crop_resize(sample, test_path, "pp/")
# 
# ext_train_y = img_extend_y(train_y)
# ext_train_y.to_csv("F:/Code/Python/3 Image ISM/train/pp/extended_train.csv")
# ext_test_y = img_extend_y(test_y)
# ext_test_y.to_csv("F:/Code/Python/3 Image ISM/test/pp/extended_test.csv")
# 
# del labels
# del sample
# del train_path
# del test_path
#==============================================================================

#Engineered Image Read-In
labels = pd.read_csv("F:/Code/Python/3 Image ISM/train/pp/extended_train.csv")
train_path = ("F:/Code/Python/3 Image ISM/train/pp/")

end_train_x = ext_readin_x(labels, train_path)
end_train_y = readin_y(labels, train_path)

end_train_x = np.int16(end_train_x)
end_train_y = np.uint8(end_train_y)

del labels
del train_path

#Train the model - VGG (modified) with Bagging
noise = 0.1 * (1.0/255)

def neuralnet():
    model = Sequential()
    model.add(GaussianNoise(noise, input_shape = (224, 224, 3)))
    
    model.add(Conv2D(32, (3, 3), activation = "relu", padding = "same"))
    model.add(MaxPooling2D((2,2), strides = (2,2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3, 3), activation = "relu", padding = "same"))
    model.add(MaxPooling2D((2,2), strides = (2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation = "relu", padding = "same"))
    model.add(MaxPooling2D((2,2), strides = (2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation = "relu", padding = "same"))
    model.add(MaxPooling2D((2,2), strides = (2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), activation = "relu", padding = "same"))
    model.add(MaxPooling2D((2,2), strides = (2,2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense((2048), activation = "relu", kernel_regularizer = regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense((512), activation = "relu", kernel_regularizer = regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(1))

    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    print(model.summary())

    return model

model = neuralnet()

splits = 10
bags = 10

for i in range(bags):
    gc.collect()

    model = neuralnet()
    
    kf = model_selection.KFold(n_splits = splits, 
                               shuffle = True
                               )

    for train_index, test_index in kf.split(end_train_x):
        X_train, X_test = end_train_x[train_index], end_train_x[test_index]
        y_train, y_test = end_train_y[train_index], end_train_y[test_index]

    gc.collect()

    y_train = np.uint8(y_train)
    y_test = np.uint8(y_test)
    train_index = np.int16(train_index)
    test_index = np.int16(test_index)

    datagen = ImageDataGenerator(featurewise_center = True,
                                 #featurewise_std_normalization = True,
                                 rotation_range = 7.5,
                                 width_shift_range = 0.05,
                                 height_shift_range = 0.05,
                                 shear_range = 0.05,
                                 zoom_range = 0.05,
                                 horizontal_flip = True
                                 )

    datagen.fit(X_train)
        
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizers.adam(),
                  metrics = ['accuracy']
                  )

    checkpointer = ModelCheckpoint(filepath = "F:/Code/Python/3 Image ISM/Models/" + str(i) + "/{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{acc:.4f}-{val_acc:.4f}.hdf5", 
                                   verbose = 0)

    
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta = 0,
                               patience = 80
                               )

    gc.collect()

    model_vgg = model.fit_generator(datagen.flow(X_train, 
                                                 y_train, 
                                                 batch_size = 16,
                                                 shuffle = True
                                                 ),
                                    validation_data = (X_test, y_test), 
                                    steps_per_epoch = len(X_train) / 16,
                                    epochs = 300, 
                                    callbacks = [early_stop, checkpointer],
                                    verbose = 1
                                    )

    #Summarize history for accuracy check
    plt.plot(model_vgg.history['acc'],'r')
    plt.plot(model_vgg.history['val_acc'],'b')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(model_vgg.history['loss'],'r')
    plt.plot(model_vgg.history['val_loss'],'b')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    gc.collect()

#Predict
sample = pd.read_csv("F:/Code/Python/3 Image ISM/test/pp/extended_test.csv")
test_path = ("F:/Code/Python/3 Image ISM/test/pp/")

end_test_x = ext_readin_x(sample, test_path)
end_test_y = readin_y(sample, test_path)

end_test_x = np.int16(end_test_x)
end_test_y = np.uint8(end_test_y)

end_test_p = bulk_predict(end_test_x)
end_test_r = pred_vote(end_test_p)

#Write
write_to_file(end_test_r)