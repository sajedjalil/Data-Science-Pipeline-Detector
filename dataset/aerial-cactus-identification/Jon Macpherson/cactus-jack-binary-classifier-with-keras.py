# Here is a list of people and projects for which credit is due. I've learned a great deal from their code:
# Angrew Ng & Coursera - https://www.coursera.org/learn/machine-learning
# Anezka Kolaceke - https://www.kaggle.com/anezka/cnn-with-keras-for-humpback-whale-id
# Peter - https://www.kaggle.com/pestipeti/keras-cnn-starter
# Yassine Ghouzam, PhD - https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
# Martin Piotte - https://www.kaggle.com/martinpiotte/bounding-box-model
# as well as countless stackoverflow posts, online tutorials and videos. 

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import PIL

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, ZeroPadding2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


#%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Binary Classification Script: 

train_df = pd.read_csv("../input/train.csv")
width = 32
height = 32
# Depth is the number of color in an image. Set to 3 for full color or 1 for grayscale images 
depth = 3 
# Number of Epochs to train, 
num_iter = 50
# The batch size for training. 
batch_size = 16


def prepareImages(data, m, dataset, i=150, j = 150, depth = 1):    
    """Import images for all image sets. 
    All training data should be stored in data['Image']
    This subroutine is only used to load the test data. 
    
    """

    print("Preparing images, graymode is ", depth)
    if depth < 3:
        X_train = np.zeros((m, i, j, 1)) 
    else: 
        X_train = np.zeros((m, i, j, 3))
    count = 0
    for fig in data['id']:
        # Read in the images from files
        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(i,j))
        if depth < 3:
            # convert to grayscale
            img = img.convert('L')      
        # Convert the image object to an numpy array 
        img = image.img_to_array(img)
        X_train[count] = img
        if (count%500 == 0):
            print("Loading image: ", count+1, ", ", fig)
        count += 1
    return X_train
    
    
def prepare_labels(y):
    # Convert the labels from names or numbers to a one-hot encoding. 
    # Not used in this binary example, but very useful for multi-class classification 
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)
    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder
    
    
def add_layer(model, filters, f_size, stride, pad):
    # I found when working on another project that I was constantly modifing the model by adding and 
    # removing layers as I tried large or small images. I added this function in combination with the 
    # build_model function below to automatically make a model somewhat optimized for the input size, which 
    # reduces the output to about somewhere between 2 x 2 x num_filters and 4 x 4 x Num filters. 
    # I've found that if you go into the dense layers with anything larger then that it seems to slow 
    # training down considerably 
    model = Conv2D(filters, (f_size,f_size), strides=(stride,stride), padding=pad, activation='relu')(model)
    model = MaxPooling2D(pool_size=2, strides=None)(model)
    model = BatchNormalization(axis=3)(model)
    # Dropout rate. .2 seems to work pretty good 
    # note this is dropout on EVERY layer 
    model = Dropout(rate=0.2)(model)   
    return model 
    
    
def build_model(input_shape, num_labels):
    mod_x_in = Input(input_shape)
    # First, Batch Normalize to speed training
    mod_X = BatchNormalization(axis=3)(mod_x_in)
    cur_size = input_shape[0]
    # The initial layers filters:, this number is increased below automatically for additional layers
    filters = 48
    # Filter size stays the same for every layer, 3 and 5 seem to work best in this script
    f_size = 5
    # set the stride here, I have not tried anything over 1. 
    stride = 1
    # padding for all layers
    pad = 'same'
    # just used to display stats about the layers
    layer = 0
    while (cur_size / 2) > 2:
        mod_X = add_layer(mod_X, filters, f_size, stride, pad)
        # after the first layer, increase the number of filters by a factor of: 
        # I've had good luck with 1.2 to 2.4 here. anything more seems a waste, and less slows training.
        filters = int(filters * 2.4)
        # used to display stats only. 
        cur_size = cur_size / 2
        # stats:
        layer += 1
        print("Layer ", layer , "filters", filters, "current window size", cur_size)
    # flatten all the data and feed it into dense layers. 
    mod_X = Flatten()(mod_X)
    # use sigmoid for binary classification, softmax for multiclass classification
    mod_X = Dense(num_labels, activation='sigmoid', name='sigmoid')(mod_X)
    print(mod_X.shape)
    model = Model(inputs=mod_x_in, outputs=mod_X, name='Cactus Finder')
    return model


#y, label_encoder = prepare_labels(train_df['has_cactus'])
y = train_df['has_cactus']
model = build_model((width, height, depth), 1)
model.summary()

# read in all the image files. 
X = prepareImages(train_df, train_df.shape[0], "train/train", width, height, depth)

# Split into train, validation sets 
random_state = 4
X, X_val, y, y_val = train_test_split(X, y, test_size = 0.1, random_state=random_state)


model.summary()
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])


# So this little gem of code takes the original data, and augments it a bit for 
# each iteration of training. T
augments = ImageDataGenerator(
        featurewise_center=False,  # input mean to 0 over the all the data
        samplewise_center=False,  # sample mean to 0
        featurewise_std_normalization=False,  # divide by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # 
        rotation_range=10,  # random rotation, 0 to 180
        zoom_range = 0.10, # Randomly zoom in 
        width_shift_range=0.10,  # random horizontally, .01 to .99, fraction of width
        height_shift_range=0.10,  # random vertical shift. .01 to .99 fraction of height
        horizontal_flip=True,  # flip images horizontal, random
        vertical_flip=True)  # flip images vertical, random 

augments.fit(X)
steps_epoch = X.shape[0] // batch_size
# Train the model with augmented data. 
history = model.fit_generator(augments.flow(X, y,batch_size=batch_size),  
        epochs=num_iter, steps_per_epoch=steps_epoch, validation_data= (X_val,y_val), verbose=2)


# Now for the test data
test = os.listdir("../input/test/test/")
col = ['id']
test_df = pd.DataFrame(test, columns=col)

test_df.head()
# load and prepare images for the test set. 
test_X = prepareImages(test_df, test_df.shape[0], "test/test/", width, height, depth)



predictions = model.predict(np.array(test_X), verbose=1)
pred_df = pd.DataFrame(predictions, columns=['has_cactus'])
pred_df['has_cactus'] = pred_df['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)

# Write out the results. Note, you *must* commit before the script will generate a submission file that 
# you can submit via a link on the output tab after the commit is complete. 
result_df = pd.concat([test_df, pred_df], axis=1, sort=False)
print(result_df.head())
result_df.to_csv('submission.csv', index=False)