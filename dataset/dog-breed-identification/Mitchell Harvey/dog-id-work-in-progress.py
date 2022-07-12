# dog breed kaggle

# copied from last notebook 
# Put these at the top of every notebook, to get automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline

# Importing required libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import sklearn
import matplotlib.pyplot as plt

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Input

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import load_img


# get data ready
y_train = pd.read_csv("/home/mitchell/kaggleData/dogBreedIdentifier/data/labels.csv")

train_image_dir = "/home/mitchell/kaggleData/dogBreedIdentifier/data/train/"
test_image_dir = "/home/mitchell/kaggleData/dogBreedIdentifier/data/test/"

df_test = pd.read_csv('/home/mitchell/kaggleData/dogBreedIdentifier/data/sample_submission.csv')


#one hot envcoding of the labesl
breed = pd.Series(y_train['breed'])
breed
one_hot = pd.get_dummies(breed, sparse = True)
#one_hot


one_hot_labels = np.asarray(one_hot)
one_hot_labels.shape


#create pivto table then plot to see the distribution of the training set
y_plot=pd.pivot_table(y_train, index='breed', aggfunc=len)
y_plot = y_plot.sort_values('id', ascending=False)
#y_plot
#y_plot.head()


#using: 'ls' > trainFilenames.csv 
# in terminal, created a csv with all filenames.

trainFilenames = pd.read_csv("/home/mitchell/kaggleData/dogBreedIdentifier/data/trainFilenames.csv", header=None)
trainFilenames.size



# this line reformats the data into numpy array
trainFilenames=trainFilenames.T.as_matrix()
trainFilenames=trainFilenames.tolist()
#trainFilenames


#list comprehension trick to take list out of a list..
trainFilenames = [y for x in trainFilenames for y in x]
#trainFilenames

# we can use this to join 

train_img_paths = [train_image_dir + s for s in trainFilenames]
#train_img_paths



# initialize lists and variables
x_train = []
train_labels = []
x_test = []
im_size = 128
img_height=im_size
img_width=im_size
bs = 32



i = 0 
for f, breed in tqdm(y_train.values):
    
    img = load_img(train_image_dir + '{}.jpg'.format(f), target_size=(img_height, img_width))
    #before image is appended it must be converted into array
    img = np.array(img, dtype="int32")
    x_train.append(img)
    
    
    label = one_hot_labels[i]
    train_labels.append(label)
    i += 1


for f in tqdm(df_test['id'].values):
    
    img = load_img(test_image_dir + '{}.jpg'.format(f), target_size=(img_height, img_width))
    img = np.array(img, dtype="int32")
    x_test.append(img)



#these few lines reformat the data
y_train_raw = np.array(train_labels, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.
x_test  = np.array(x_test, np.float32) / 255.


#x_train_raw_min = x_train_raw[:9000,:,:,:]


print(x_train_raw.shape)
print(y_train_raw.shape)
print(x_test.shape)


#We can see above that there are 120 different breeds. 
#We can put this in a num_class variable below that can then be used when creating the CNN model.
num_class = y_train_raw.shape[1]


# build a croos val set
X_train, X_val, Y_train, Y_val = train_test_split(x_train_raw, y_train_raw, test_size=0.2, random_state=0)


# Initialising the CNN
classifier = ResNet50(include_top=False, weights='imagenet', 
                   input_shape=(img_width,img_height, 3))

#Add a layer where input is the output of the  second last layer 
x = Flatten(name='flatten')(classifier.output)
#x = Dense(num_classes, activation='softmax', name='predictions')(x)
x = BatchNormalization()(x)
x = Dropout(0.8)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.8)(x)
x = Dense(120, activation='softmax')(x)

#Then create the corresponding model 
my_model = Model(inputs=classifier.input, outputs=x)
my_model.summary()





#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
my_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


my_model.fit(X_train, Y_train, epochs=20,batch_size=bs, validation_data=(X_val, Y_val), verbose=1)