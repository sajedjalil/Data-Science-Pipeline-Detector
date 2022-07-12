############## ALL IMPRTS ########################
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import History, ModelCheckpoint, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import pandas as pd

############## ALL USER FUNCTIONS ########################

#Used to save model and check training progress
def plot_progress(epoch,logs):
    plt.figure()
    plt.plot(range(epoch+1),history.history['loss'],'b',label='trainin loss')
    plt.plot(range(epoch+1),history.history['val_loss'],'r',label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('training error')
    plt.legend(loc='best')
    plt.savefig(locpath+'training_error.png')
    plt.close('all')

############## TRAINING SETTINGS ########################

K.set_image_dim_ordering('th') # For image align

batch_size = 32
nb_epoch = 50
nb_classes = 8

locpath = "your local folder"
filepath = locpath+"net.{epoch:02d}-{val_loss:.2f}.h5"

############## LOAD TRAINING DATA ########################

train_datagen = ImageDataGenerator(
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                '../data/train',  # this is the target directory
                target_size=(256, 256),  # all images will be resized to 224x224
                batch_size=batch_size,
                class_mode='categorical')  # since there are 8 different fish labels

validation_generator = test_datagen.flow_from_directory(
                '../data/val',
                target_size=(256,256),
                batch_size=batch_size,
                class_mode='categorical')

############## DEFINE MODEL ########################

net = Sequential()
net.add(Convolution2D(32, 3, 3, input_shape=(3, 256, 256)))
net.add(Activation('relu'))
net.add(MaxPooling2D(pool_size=(2, 2)))

net.add(Convolution2D(32, 3, 3))
net.add(Activation('relu'))
net.add(MaxPooling2D(pool_size=(2, 2)))

net.add(Convolution2D(64, 3, 3))
net.add(Activation('relu'))
net.add(MaxPooling2D(pool_size=(2, 2)))

net.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
net.add(Dense(64))
net.add(Activation('relu'))
net.add(Dropout(0.3))
net.add(Dense(nb_classes))
net.add(Activation('softmax'))

opt = RMSprop()

net.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

history = History()
plot_progress_cb = LambdaCallback(on_epoch_end=plot_progress)
check = ModelCheckpoint(filepath=filepath,monitor='val_acc', verbose=1,
                        save_best_only=True)

print("Starting training...")

net.fit_generator(
            train_generator,
            samples_per_epoch=5000,
            nb_epoch=nb_epoch,
            verbose=1,
            callbacks=[history,check,plot_progress_cb],
            validation_data=validation_generator,
            nb_val_samples=1000)

print("Training end...")

####################################
#To run after training 
####################################
############## MY FUNCTIONS ########################
def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x

############## TRAINING SETTINGS ########################

#K.set_image_dim_ordering('th') # For image align

batch_size = 32
nb_epoch = 50
nb_classes = 8
nb_rows = 1000
classes = ('ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT') 
nm_img = ('image')

testimpath = "yourLocalFolder/test_stg1/"
locpath = "yourLocalFolder/trainingResults/"
#Saved during training
netpath = locpath+"net.23-0.41.h5"


############## LOAD MODEL ########################
net = load_model(netpath)

############## PREDICT ########################

test_imgs = os.listdir(testimpath)
predict_df = pd.DataFrame(index=np.arange(0, nb_rows), columns=classes)

count = 0
for img in test_imgs:
    img = image.load_img(testimpath+img, target_size=(256, 256))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #Actual prediction
    preds = net.predict_proba(x)[0]
    predict_df.loc[count] = preds
    count += 1
    
img_name_df = pd.read_csv("../submission/sample_submission_stg1.csv")
img_name_df = img_name_df[nm_img]

predict_df.insert(0,nm_img,img_name_df)

predict_df.to_csv('nature_predicts.csv',index=False)
