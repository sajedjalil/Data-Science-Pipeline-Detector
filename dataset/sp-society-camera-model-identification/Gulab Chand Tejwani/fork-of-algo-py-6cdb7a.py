#This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.image as mpimg       # reading images to numpy arrays
import matplotlib.pyplot as plt        # to plot any graph
import matplotlib.patches as mpatches  # to draw a circle at the mean contour
from skimage.data import imread
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn import metrics
import cv2
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# To calculate a normalized histogram 


from skimage import measure            # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality


# matplotlib setup
#%matplotlib inline
from pylab import rcParams
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from skimage.feature import local_binary_pattern
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from pathlib import Path
input_path = Path('../input')
train_path = input_path / 'train'
test_path = input_path / 'test'
#train_path = "C:\\Users\\Vohra\\Downloads\\all\\train\\train\\"

cameras = os.listdir(train_path)
print(cameras)

train_images = []
for camera in cameras:
    for fname in sorted(os.listdir(train_path / camera)):
        train_images.append((camera, fname))

train = pd.DataFrame(train_images, columns=['camera', 'fname'])
#print(train)
print(train.shape)
print(train.values)


#test_images = []
#for fname in sorted(os.listdir(test_path)):
#    test_images.append(fname)
#
#test = pd.DataFrame(test_images, columns=['fname'])
#print(test.shape)

nt = 20
train_images2 = train_images[:nt]
train_images2 = train_images2 + train_images[1000:1000+nt] 
train_images = train_images2
#test_images = test_images[:nt]

X_train = []
Y_train = []

for i in train_images:
    i_path = '../input/train/' + i[0] + '/' + i[1];
    img_aux = cv2.imread(i_path)
    print(i)
    #img_aux = np.array(img_aux, dtype=np.uint8)
    train_image = i_path
    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(img_aux, cv2.COLOR_BGR2GRAY)
    radius = 3
    # Number of points to be considered as neighbourers 
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    # Calculate the histogram
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    # Append image path in X_name
    # Append histogram to X_name
    X_train.append(hist)
    # Append class label in y_test
    Y_train.append(i[0])
    
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.25, random_state=0)
print(x_test)
print(y_test)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)


## Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
score = classifier.score(x_test, y_test)
print(score)

#svm
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(x_train, y_train)
y_pred1 = svclassifier.predict(x_test)  
print(y_pred1)
## Use score method to get accuracy of model
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
score = classifier.score(x_test, y_test)
print(score)
#print(confusion_matrix(np.argmax(y_test), y_pred))

#output = pd.DataFrame(columns=['f_name', 'camera'])
#output['f_name'] = list(y_test)
#output['camera'] = y_pred


import os,cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop,adam
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import pandas as pd

from pathlib import Path
input_path = Path('../input')
train_path = input_path / 'train'
#test_path = input_path / 'test'
#train_path = "C:\\Users\\Vohra\\Downloads\\all\\train\\train\\"
test_path = input_path / 'test'
data_dir_list = os.listdir(train_path)
print(data_dir_list)

#cameras = os.listdir(train_path)

#train_images = []
#for camera in cameras:
 #   for fname in sorted(os.listdir(train_path / camera)):
  #      train_images.append((camera, fname))

#train = pd.DataFrame(train_images, columns=['camera', 'fname'])
#print(train.shape)
#print(train)

#test_images = []
#for fname in sorted(os.listdir(test_path)):
 #   test_images.append(fname)

#test = pd.DataFrame(test_images, columns=['fname'])
#print(test.shape)

#nt = 20
#train_images2 = train_images[:nt]
#train_images2 = train_images2 + train_images[1000:1000+nt] 
#train_images = train_images2
#test_images = test_images[:nt]


img_rows=128
img_cols=128
num_channel=1
num_epoch=20

# Define the number of classes
num_classes = 10

labels_name={'HTC-1-M7':0,'iPhone-4s':1,'iPhone-6':2,'LG-Nexus-5x':3,'Motorola-Droid-Maxx':4,'Motorola-Nexus-6':5,'Motorola-X':6,'Samsung-Galaxy-Note3':7,'Samsung-Galaxy-S4':8,'Sony-NEX-7':9}

print(labels_name)

img_data_list=[]
labels_list = []

for dataset in data_dir_list:
	img_list=os.listdir(train_path/ dataset)
	print ('Loading the images of dataset-'+'{}\n'.format(dataset))
	label = labels_name[dataset]
	for img in img_list:
		input_img=cv2.imread('../input/train' + '/'+ dataset + '/'+ img )
		print(img)
		input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(128,128))
		img_data_list.append(input_img_resize)
		labels_list.append(label)
		
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print ('jdjd',img_data.shape)

labels = np.array(labels_list)
# print the count of number of samples for different classes
print(np.unique(labels,return_counts=True))
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)

# Defining the model
input_shape=img_data[0].shape
print('hhh',input_shape)

model = Sequential()

model.add(Convolution2D(32, 3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

X_train=X_train.reshape([-1,128, 128,1])
X_test=X_test.reshape([-1,128, 128,1])
print("gulab")
print(X_test.shape)
print(X_train.shape)

hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))
print(hist)

from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
#print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)


#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names =['class 0(HTC-1-M7)','class 1(iPhone-4s)','class 2(iPhone-6)','class 3(LG-Nexus-5x)','class 4(Motorola-Droid-Maxx)','class 5(Motorola-Nexus-6)','class 6(Motorola-X)','class 7(Samsung-Galaxy-Note3)','class 8(Samsung-Galaxy-S4)','class 9(Sony-NEX-7)']					
predicts = [target_names[p] for p in y_pred]
#print(predicts)
#print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

## Use score method to get accuracy of model
score = model.evaluate(X_test, y_test)
print(score)
scores = model.evaluate(X_test,y_test, verbose=0)
print("loss",scores[0])
print("acc",scores[1])


df = pd.DataFrame(columns=['fname', 'camera'])
df['fname'] = list(y_test)
df['camera'] = predicts
df.to_csv("sub.csv", index=False)
					