#### A small CNN model to get you started. 
#### Feel free to play with this model
#### and improve results

# Import dependencies
import csv
import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K

# Set parameters
DIR = '../input/'
rows = 28 
cols = 28
channels = 1
pixels = cols * rows
classes = 10
trainImages = 42000
testImages = 28000
epoch = 10

# Initialize training dataset
yTrain = np.zeros((trainImages))
xTrain = np.zeros((trainImages, channels, cols, rows))
counter = 0
skip = True

# Read in training data
trainFile = open(DIR + 'train.csv')
csv_file = csv.reader(trainFile)
for row in csv_file:
	if (skip == True):
		skip = False
		continue
	yTrain[counter] = row[0]
	temp = np.zeros((1, pixels))
	for num in range(1, pixels):
		temp[0,num - 1] = row[num]
	temp = (temp - np.mean(temp))/(np.max(temp) - np.min(temp))
	temp = np.reshape(temp, (rows, cols))
	xTrain[counter,0,:,:] = temp
	counter = counter + 1

# Initialize test dataset
yTest = np.zeros((testImages))
xTest = np.zeros((testImages, channels, cols, rows))
skip2 = True
counter2 = 0

# Read in testing data
testFile = open(DIR + 'test.csv')
csv_file2 = csv.reader(testFile)
for row in csv_file2:
	if (skip2 == True):
		skip2 = False
		continue
	yTest[counter2] = row[0]
	temp = np.zeros((1, pixels))
	for num in range(1, pixels):
		temp[0,num - 1] = row[num]
	temp = (temp - np.mean(temp))/(np.max(temp) - np.min(temp))
	temp = np.reshape(temp, (rows, cols))
	xTest[counter2,0,:,:] = temp
	counter2 = counter2 + 1

# Convert class vectors to binary class matrices
yTrain = np_utils.to_categorical(yTrain, classes)
yTest = np_utils.to_categorical(yTest, classes)

# Create model architecture
model = Sequential()
model.add(Conv2D(64, 3, 3, border_mode='valid', input_shape=(channels, rows, cols)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, 3, 3, border_mode='valid'))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('elu'))
model.add(Dense(classes))
model.add(Activation('softmax'))

# Train the model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy', 'mse'])
model.fit(xTrain, yTrain, batch_size=32, nb_epoch=epoch,validation_data=(xTest, yTest),shuffle=True)

results = np.zeros((testImages,2))
for num in range(1, testImages + 1):	
	results[num - 1,0] = num

# Predict classes and store in results
temp = model.predict_classes(xTest, batch_size=32, verbose=1)
for num in range(0, testImages):	
	results[num,1] = temp[num]
    
# Save results in csv file
results = pd.np.array(results) 
df = pd.DataFrame(results, columns = ["ImageId", "Label"])
df = df.astype('int64')
df.to_csv('submission.csv', index=False)