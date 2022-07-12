'''
Created on Nov 18, 2017

@author: karavi01
'''
import pandas as pd
import io
import cv2
import numpy as np
import tensorflow
import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from subprocess import check_output
from copyreg import pickle
print(check_output(["ls", "../input"]).decode("utf8"))

import bson
import matplotlib.pyplot as plot

from skimage.data import imread
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

categories = pd.read_csv('../input/category_names.csv', index_col='category_id')
data = bson.decode_file_iter(open('../input/train.bson','rb'))
batchSize = 31
dimData = 180*180
def getCategories(data):
    productIDArray = []
    categoryIDArray = []
    productIndex = []

    try:
        for c,d in enumerate(data):            
            productIndex.append(c)
            product_id = d['_id']
            category_id = d['category_id']        
            product_imgs = d['imgs']                           
            categoryIDArray.append(category_id)
            productIDArray.append(product_id)
            
    except IndexError:
        print('error reading data')
    
    productCategory_dict = {'product_id':productIDArray,'category_id':categoryIDArray}
    productCategory_df = pd.DataFrame(productCategory_dict,columns=['product_id','category_id'],index=productIndex)
    return productCategory_df

def get_data_generator(data_id,labelencoder,numLabels,dimData,batchSize = 31):             
    while True:
        data = bson.decode_file_iter(open('../input/train.bson','rb'))
        allEdges = np.empty((batchSize,dimData))
        allCategoryID = np.empty((batchSize,numLabels))
        i = 0                        
        for c,d in enumerate(data):
            product_id = d['_id']
            category_id = d['category_id']            
            if product_id in data_id:
                pic = d['imgs'][0]
                picture = imread(io.BytesIO(pic['picture'])) 
                edges = cv2.Canny(picture,50,200)
                edges = np.reshape(edges,(dimData,))
                allEdges[i,:] = edges
                tmpCat = labelencoder.transform([category_id])
                one_hot_labels = keras.utils.to_categorical(tmpCat[0], num_classes=numLabels)
                allCategoryID[i] = one_hot_labels
                i = i + 1
                if i >= batchSize:                    
                    yield allEdges,allCategoryID
                    allEdges = np.empty((batchSize,dimData))
                    allCategoryID = np.empty((batchSize,numLabels))   
                    i = 0    
        if i>0:
            allEdges = allEdges[:i-1,:]
            allCategoryID = allCategoryID[:i-1]
            yield allEdges, allCategoryID

productCategory_df = getCategories(data)
train_val_id,test_id,train_val_lab,test_lab = train_test_split(productCategory_df['product_id'].values,productCategory_df['category_id'].values,test_size = 0.1)
train_id,val_id,train_lab,val_lab = train_test_split(train_val_id,train_val_lab,test_size = 0.1)
numLabels = len(pd.unique(productCategory_df['category_id']))
labelencoder = LabelEncoder()
labelencoder.fit(productCategory_df['category_id']) 

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(numLabels, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(get_data_generator(train_id, labelencoder, numLabels,dimData, batchSize),
                     epochs = 7, verbose = True,steps_per_epoch = 1000,
                    validation_data = get_data_generator(val_id, labelencoder,
                                                          numLabels,dimData, batchSize),validation_steps = 5)
metric_values = model.evaluate_generator(get_data_generator(test_id,labelencoder,numLabels,dimData, batchSize),
                         steps=20)
print("## Test results = {}".format(str(metric_values)))