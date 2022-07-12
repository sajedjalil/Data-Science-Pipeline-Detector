from keras import backend as K
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
from keras.utils import np_utils
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import cv2
import os.path


# Code to read and preprocess Images
def preprocess_input(i):
    i = np.divide(i, 255.0)
    i = np.subtract(i, 1.0)
    i = np.multiply(i, 2.0)
    return i

def get_processed_image(img_path):
    # Load image and convert from BGR to RGB
    im = np.asarray(cv2.imread(img_path))[:, :, ::-1]
    im = cv2.resize(im, (299, 299))
    im = preprocess_input(im)
    if K.image_dim_ordering() == "th":
        im = np.transpose(im, (2,0,1))
        im = im.reshape(-1, 3, 299, 299)
    else:
        im = im.reshape(-1,299, 299, 3)
    return im


train_I = [] # Defining an empty array to read training images into
train_size = 0 

# Reding the Labels CSV
train_L = pd.read_csv(r'C:\Users\Dhanu\PycharmProjects\Invasive_Species\train_labels.csv')

imgfldr = 'C:\\Users\\Dhanu\\PycharmProjects\\Invasive_Species\\train\\'

# Reading training images into the list corresponding to the Name column in train_L
for i in range(len(train_L)):
    img = get_processed_image( imgfldr + str(train_L.ix[i][0]) +'.jpg' )
    train_I.extend(img)
    train_size = train_size + 1

# Dropping Name column from the train_labels CSV
train_L.drop(train_L.columns[[0]], axis=1, inplace=True)

print(train_size)

# Randomly shuffling the data
train_I, train_L = shuffle(train_I, train_L, random_state=3)

# Data Augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(train_I)

# Splitting the data into Train & Validation sets
valid_I = train_I[1840:]
valid_L = train_L[1840:]

train_I = train_I[0:1840]
train_L = train_L[0:1840]

# Defining the base model
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

# Fine tuning the top layer
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)

# creating the final model
model = Model(input=base_model.input, output=predictions)

# Freezing the layers of pretrained model
for layer in base_model.layers:
    layer.trainable = False

# compile the model
model.compile(loss="binary_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])

model.fit_generator(datagen.flow(np.array(train_I), np.array(train_L), batch_size=16), steps_per_epoch=115, epochs=5,
                    verbose=1, validation_data=datagen.flow(np.array(valid_I), np.array(valid_L), batch_size=16),
                    validation_steps=((len(valid_L))/16))

model.save('Invasive_Species_3.h5')


# I splitted the Training and Prediction in a way to avoid Memory error 
# Below Code can be run separetly along with 'code to read and preprocess images'


test_I = [] # Defining an empty array to read training images into
test_size = 0

# Reding Sample submission CSV into a dataframe
test_L = pd.read_csv(r'C:\Users\Dhanu\PycharmProjects\Invasive_Species\sample_submission.csv')

imgfldr = 'C:\\Users\\Dhanu\\PycharmProjects\\Invasive_Species\\test\\'

# Reading test images into the list corresponding to the Name column in test_L
for i in range(len(test_L)):
    img = get_processed_image( imgfldr + str(int(test_L.ix[i][0])) +'.jpg' )
    test_I.extend(img)
    test_size = test_size + 1

print(test_size)

# Notice no shuffling & data augmentation for test images

# Loading the saved model
model = load_model('C:\\Users\\Dhanu\\PycharmProjects\\Invasive_Species\\Invasive_Species_3.h5')

# compile the model
model.compile(loss="binary_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


# Predicting the image label
# I predicted for 100 images at a time (In a way to avoid crashing)   

Curr_ID = 0
Step_Size = 100

Steps = int(test_size/Step_Size) # Calculating no.of iterations required @ given step size (15 in this case for 1500 images)
final_step = test_size%Step_Size # Calatating the no. of remaining images after last iteration (31 in this case)

# Iterating over images to predict labels
for i in range (Steps):
        T_predict = model.predict(np.array(test_I[Curr_ID:(Curr_ID + Step_Size)]))
        # Writing over the predicted labels in Dtaframe
        test_L['invasive'][Curr_ID:(Curr_ID + Step_Size)] = np.concatenate(T_predict)
        Curr_ID = Curr_ID + Step_Size

# Predicting labels of remaining images
T_predict = model.predict(np.array(test_I[Curr_ID:(Curr_ID + final_step)]))
test_L['invasive'][Curr_ID:(Curr_ID + final_step)] = np.concatenate(T_predict)


##### Code to predict for all test images at once
##  T_predict = model.predict(np.array(test_I))
##  test_L['invasive'] = np.concatenate(T_predict)
#####

# Saving the predicted labels from dataframe to a CSV
test_L.to_csv('C:\\Users\\Dhanu\\PycharmProjects\\Invasive_Species\\submission.csv', index=None)
