
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

from tqdm import tqdm

train_images = glob("train/*jpg")
test_images = glob("test/*jpg")
df = pd.read_csv("train.csv")

df["Image"] = df["Image"].map( lambda x : "train/"+x)
ImageToLabelDict = dict( zip( df["Image"], df["Id"]))

SIZE = 100
#image are imported and resized
def ImportImage( filename):
    img = Image.open(filename).resize( (SIZE,SIZE))
    img = np.array(img)
    if img.ndim == 2: #imported BW picture and converting to "dumb RGB"
        img = np.tile( img, (3,1,1)).transpose((1,2,0))
    return img
x_train = np.array([ImportImage( img) for img in train_images],dtype=np.uint8)



print( "%d training images" %x_train.shape[0])

print( "Nbr of samples/class\tNbr of classes")
for index, val in df["Id"].value_counts().value_counts().sort_index().iteritems():
    print( "%d\t\t\t%d" %(index,val))




class LabelOneHotEncoder():
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()
    def fit_transform(self, x):
        features = self.le.fit_transform( x)
        return self.ohe.fit_transform( features.reshape(-1,1))
    def transform( self, x):
        return self.ohe.transform( self.la.transform( x.reshape(-1,1)))
    def inverse_tranform( self, x):
        return self.le.inverse_transform( self.ohe.inverse_tranform( x))
    def inverse_labels( self, x):
        return self.le.inverse_transform( x)

y = list(map(ImageToLabelDict.get, train_images))
lohe = LabelOneHotEncoder()
y_cat = lohe.fit_transform(y)


#use of an image generator for preprocessing and data augmentation
x_train = x_train.reshape( (-1,SIZE,SIZE,3))
input_shape = x_train[0].shape
#x_train = x_train.astype("float32")
y_train = y_cat

image_gen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
	rescale=1./255,
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True)

#training the image preprocessing
image_gen.fit(x_train, augment=True)

#visualization of some images out of the preprocessing
#augmented_images, _ = next( image_gen.flow( x_train, y_train.toarray(), batch_size=4*4))
#plotImages( augmented_images)




batch_size = 16
num_classes = len(y_cat.toarray()[0])
epochs = 10 #x_train.shape[0]//batch_size + 1

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

model = Sequential()

#picking vgg16 as pretrained (base) model https://keras.io/applications/#vgg16
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
for layer in conv_base.layers:
    layer.trainable = False

#maybe unfreeze last layer
conv_base.layers[-2].trainable = True

model.add( conv_base)
model.add(Flatten())
model.add(Dropout(0.33))
model.add(Dense(48, activation='relu')) #64
model.add(Dropout(0.33))
model.add(Dense(48, activation='relu')) #48
model.add(Dropout(0.33))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()
model.fit_generator(image_gen.flow(x_train, y_train.toarray(), batch_size=batch_size),
          steps_per_epoch=x_train.shape[0] // epochs,
          epochs=epochs,
         verbose=1)




import gc
del x_train, y_train
gc.collect()

import warnings
from os.path import split

print( "Exporting predictions..")
with open("sample_submission.csv","w") as f:
    with warnings.catch_warnings():
        f.write("Image,Id\n")
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        for image in tqdm(test_images):
            img = ImportImage( image)
            x = img.astype( "float32")
            #applying preprocessing to test images
            x = image_gen.standardize( x.reshape(1,SIZE,SIZE,3))
            y = model.predict_proba(x.reshape(1,SIZE,SIZE,3))
            predicted_args = np.argsort(y)[0][::-1][:5]
            predicted_tags = lohe.inverse_labels( predicted_args)
            image = split(image)[-1]
            predicted_tags = " ".join( predicted_tags)
            f.write("%s,%s\n" %(image, predicted_tags))


