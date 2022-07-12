import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize, rotate, SimilarityTransform, warp
from skimage.filters import sobel
import glob

import gc
gc.collect()

np.random.seed(1337) # for reproducibility

# Loading the train.csv to find features for each training point
train = pd.read_csv('../input/train.csv', usecols=['id', 'species'])
mtrain = train.shape[0]
test = pd.read_csv('../input/test.csv', usecols = [0])
mtest = test.shape[0]
#print(test.head())

# Resize and read images
# Fill image to same size
print('Loading and resizing images...')# % len(image_paths))
img_rows, img_cols = 64, 64 #96,96 #40, 40
output_shape = (img_rows, img_cols)

train_images = np.zeros((mtrain, img_rows, img_cols))
for i in range(mtrain):
    #image = imread('images/'+str(train.id.iloc[i])+'.jpg')
    image = imread('../input/images/'+str(train.id.iloc[i])+'.jpg')
    rimage = resize(image, output_shape=output_shape)
    train_images[i] = sobel(rimage)
    
test_images = np.zeros((mtest, img_rows, img_cols))
for i in range(mtest):
    image = imread('../input/images/'+str(test.id.iloc[i])+'.jpg')
    rimage = resize(image, output_shape=output_shape)
    test_images[i] = sobel(rimage)

print('Train images shape: {}'.format(train_images.shape)) # 990
print('Test images shape: {}'.format(test_images.shape)) # 594

# Target
le = LabelEncoder()
target = le.fit_transform(train.species)
print("Target shape: {}".format(target.shape))
print(np.unique(target).size)
print(target[:5])

# Scale training data
#scaler = StandardScaler().fit(train_images)
#train_images = scaler.transform(train_images)

# Create random train and validation sets out of 20% samples
#Xtrain, Xval, ytrain, yval = train_test_split(train_images, target, test_size=0.15,
#                                           stratify=target, random_state=14) #11)
Xtrain, ytrain = train_images, target
Xval, yval = Xtrain, ytrain # Bigger validation set
print('\nXtrain, ytrain shapes ' + str((Xtrain.shape, ytrain.shape)))
print('Xval, yval shapes ' + str((Xval.shape, yval.shape)))

# Reshape data as a 4-dim tensor [number samples, width, height, color channels]
print('Reshape as 4-dim tensor (Tensorflow backend)')
Xtrain = Xtrain.reshape(Xtrain.shape[0], img_rows, img_cols, 1) # 792
Xval = Xval.reshape(Xval.shape[0], img_rows, img_cols, 1) # 198
Xtest = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
print(Xtrain.shape, Xval.shape, Xtest.shape)

# Image batch generator
def imageGenerator(X, y, batch_size):
    img_rows, img_cols = X.shape[1], X.shape[2]
    resc = 0.02
    rot = 5
    transl = 0.01*img_rows
    while 1: # Infinite loop
        batchX = np.zeros((batch_size, img_rows, img_cols, 1))
        # batch_size random indices over train images
        batch_ids = np.random.choice(X.shape[0], batch_size)
        for j in range(batch_ids.shape[0]): # Loop over random images
            # Rotate around center
            imagej = rotate(X[batch_ids[j]], angle =rot*np.random.randn())
            # Rescale and translate
            tf = SimilarityTransform(scale = 1 + resc*np.random.randn(1,2)[0],
                                translation = transl*np.random.randn(1,2)[0]) 
            batchX[j] = warp(imagej, tf)
        #batchX = np.reshape(batchX, (batch_size,-1)) # Flattened images for FNN
        #print(batchX.shape, y[batch_ids].shape)
        yield (batchX, y[batch_ids])
        
# Convolutional Neural Network
print('Train convolutional neural network')
# Parameters
# Batch size: number of training examples in one forward/backward pass. 
# The higher the batch size, the more memory space you'll need
batch_size = 32
# Epoch: one forward pass and one backward pass of all the training examples
nb_epoch = 10
n_extension = 10
samples_per_epoch = batch_size*(n_extension*Xtrain.shape[0] // batch_size)
# number of convolutional filters to use (a hidden layer is segmented into feature maps
# where each unit in a feature map looks for the same feature but at different positions of the input image)
#nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
# Number of classes
nb_classes = np.unique(ytrain).size
#kernel_size = (5, 5)
print('Model Parameters:')
print('Batch size: %d, epochs: %d, samples per epoch: %d, n classes: %d' % 
                                (batch_size, nb_epoch,samples_per_epoch, nb_classes))

print('\nTraining Keras Convolutional Neural Network...')
# Convert class vectors to binary class matrices (one-hot encoder)
ytrain = np_utils.to_categorical(ytrain, nb_classes)
yval = np_utils.to_categorical(yval, nb_classes)
# Create model
model = Sequential()
# Add hidden layers
# Conv2D layer with 5x5 kernels (local weights) and 32 conv filters 
# (or feature maps), expects 2d images as inputs
model.add(Convolution2D(16, 5, 5,  border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5)) # Regularization method, exclude 50% units
# Another conv2D layer
model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))
# Pool2D layer, a form of non-linear down-sampling to prevent
# overfitting and provide a form of translation invariance
model.add(MaxPooling2D(pool_size=pool_size))
#model.add(Dropout(0.25)) # Regularization method, exclude 25% units
# Flattenig layer, converts 2D matrix into vectors
model.add(Flatten())
# Standard fully connected layer with 128 units
# model.add(Dense(256))
# model.add(Dropout(0.25)) # Regularization method, exclude 25% units
# model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))

# Output layer
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# Fit model with generator
model.fit_generator(imageGenerator(Xtrain, ytrain, batch_size), 
                    samples_per_epoch = samples_per_epoch,
                    nb_epoch=nb_epoch, verbose=1, validation_data=(Xval, yval))
#model.fit(Xtrain, ytrain, batch_size=batch_size, nb_epoch=nb_epoch,
#          verbose=1, validation_data=(Xval, yval))
score = model.evaluate(Xval, yval, verbose=0)
print('Validation loss: %0.5f' % score[0])
print('Validation accuracy: %0.2f' % (100*score[1]))
# 1 epoch, val 0.76, LB 1.82
# 10 epoch, val 0.03, LB 1.67
# 10 epoch (dropout 0.5), val 0.044, LB 1.43 
# 10 epoch (conv 16, dropout 0.5), val 0.033, LB 1.73

# Test predictions
print('Test predictions...')
ids = test.id
preds = model.predict_proba(Xtest)
# Submit
submission = pd.DataFrame(preds,index=ids,columns=le.classes_)
submission.to_csv('Leaf_Keras_CNN_imagegen.csv')