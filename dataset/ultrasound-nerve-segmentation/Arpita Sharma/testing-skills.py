import os
os.environ["THEANO_FLAGS"] = "base_compiledir=./some_dir_where_you_can_write/"
import theano
import numpy as np
import cv2
import glob
import scipy
from scipy import *
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import np_utils
from sklearn.metrics import log_loss


def image_preprocess(image, img_rows, img_cols):
	img = cv2.imread(image, 0)
	img_resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
	return img_resized

def create_train_data(img_rows, img_cols):
    images = glob.glob("../input/train/*[0-9].tif")
    imgs = []
    imgs_id = []
    mask = []
    a=img_rows
    b=img_cols
    
    print('Reading training images...')
    for image in images:
        flbase = os.path.basename(image)
        img = image_preprocess(image, a, b)
        img_np = np.array(img)
        imgs.append(img_np)
        imgs_id.append(flbase[:-4])
        
        mask_path = "../input/train/" + flbase[:-4] + "_mask.tif"
        maska = image_preprocess(image, img_rows, img_cols)
        mask_np = np.array(maska)
        mask.append(mask_np)
		
    print('Reading train images done.')
    return imgs, imgs_id, mask

def create_test_data(img_rows, img_cols):
    files = glob.glob("../input/test/*[0-9].tif")
    imgs = []
    imgs_id = []
    
    print ('Reading test images')
    for image in files:
        flbase = os.path.basename(image)
        img = image_preprocess(image, img_rows, img_cols)
        img_np = np.array(img)
        imgs.append(img_np)
        imgs_id.append(flbase[:-4])
        
    print('Reading test images done.')
    return imgs, imgs_id
    

def get_unet(img_rows, img_cols):
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy')

    return model
    
def preprocess(imgs, img_rows, img_cols):
    imgs_p = np.ndarray((imgs[0], imgs[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def train_and_predict(imgs_train, imgs_mask, imgs_test):
    rows = 64
    cols = 80
    
    imgs_train = np.array(imgs_train, dtype=uint8)
    imgs_mask = np.array(imgs_mask, dtype=uint8)
    imgs_test = np.array(imgs_test, dtype=uint8)
    #print (imgs_train)
    #imgs_train = preprocess(imgs_train, rows, cols)
    #imgs_mask = preprocess(imgs_mask, rows, cols)
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    print (mean)

    imgs_train -= mean
    imgs_train /= std

    imgs_mask = imgs_mask.astype('float32')
    imgs_mask /= 255.  # scale masks to [0, 1]

    print('Creating and compiling model...')
    model = get_unet(rows, cols)
    #model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    #plot(model, to_file='model.png')
    print('Fitting model...')
    #model.fit(imgs_train, imgs_mask, batch_size=32, nb_epoch=20, verbose=1, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0)])

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    
    print('Predicting masks on test data...')
    print (imgs_mask)
    #result = Image.fromstring('L', (imgs_mask.shape[1], imgs_mask.shape[0]), imgs_mask.tostring())
    #PIL.Image.show(result)
'''
    
    result = []
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    #print (imgs_mask_test)
    result = Image.fromstring('L', (imgs_mask_test.shape[1], imgs_mask_test.shape[0]), imgs_mask_test.tostring())
'''
def calling_function():
    img_rows = 64
    img_cols = 80
    imgs_train, imgs_train_id, imgs_mask = create_train_data(img_rows, img_cols)
    imgs_test, imgs_test_id = create_test_data(img_rows, img_cols)
    train_and_predict(imgs_train, imgs_mask, imgs_test)

if __name__ == '__main__':
    calling_function()