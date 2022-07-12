import numpy as np 
import pandas as pd 
import os
import cv2
from tqdm import tqdm

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import optimizers
from keras import backend as K
from keras.optimizers import SGD
from keras.layers.merge import concatenate

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split


x_train = []
x_test = []
y_train = []

path = ""
name = "Unet"
weights_path = path + name + '.h5'

df_train = pd.read_csv('../input/train_v2.csv')
df_test = pd.read_csv('../input/sample_submission_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

labels = ['blow_down',
 'bare_ground',
 'conventional_mine',
 'blooming',
 'cultivation',
 'artisinal_mine',
 'haze',
 'primary',
 'slash_burn',
 'habitation',
 'clear',
 'road',
 'selective_logging',
 'partly_cloudy',
 'agriculture',
 'water',
 'cloudy']

label_map = {'agriculture': 14,
 'artisinal_mine': 5,
 'bare_ground': 1,
 'blooming': 3,
 'blow_down': 0,
 'clear': 10,
 'cloudy': 16,
 'conventional_mine': 2,
 'cultivation': 4,
 'habitation': 9,
 'haze': 6,
 'partly_cloudy': 13,
 'primary': 7,
 'road': 11,
 'selective_logging': 12,
 'slash_burn': 8,
 'water': 15}

img_size = 64
channels = 4 #4 for tiff, 3 for jpeg

for f, tags in tqdm(df_test.values, miniters=1000):
    img = cv2.imread('../input/test-tif-v2/{}.tif'.format(f), -1)
    x_test.append(cv2.resize(img, (img_size, img_size)))
x_test  = np.array(x_test, np.float32)/255. 

for f, tags in tqdm(df_train.values, miniters=1000):
    #https://stackoverflow.com/questions/37512119/resize-transparent-image-in-opencv-python-cv2
    #If you load a 4 channel image, the flag -1 indicates that the image is loaded unchanged, so you can load and split all 4 channels directly.
    img = cv2.imread('../input/train-tif-v2/{}.tif'.format(f), -1)#0-1 voir au dessus les 2 comments
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (img_size, img_size)))
    y_train.append(targets)
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32)/255.


print(x_train.shape)
print(y_train.shape)

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2)   

print('Split train: ', len(X_train), len(Y_train))
print('Split valid: ', len(X_val), len(Y_val))


def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)
    
def get_unet(n_ch,patch_height,patch_width):
    concat_axis = 3

    inputs = Input((patch_height, patch_width, n_ch))
    
    conv1 = Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="relu", data_format="channels_last")(inputs)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool2)
    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool3)
    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool4)
    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)

    up_conv5 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv4)
    up6   = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(up6)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv6)

    up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv3)
    up7   = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(up7)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv7)

    up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv2)
    up8   = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up8)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv8)

    up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv1)
    up9   = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(up9)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv9)

    #ch, cw = get_crop_shape(inputs, conv9)
    #conv9  = ZeroPadding2D(padding=(ch[0],cw[0]), data_format="channels_last")(conv9)
    #conv10 = Conv2D(1, (1, 1), data_format="channels_last", activation="sigmoid")(conv9)
    
    flatten =  Flatten()(conv9)
    Dense1 = Dense(512, activation='relu')(flatten)
    BN =BatchNormalization() (Dense1)
    Dense2 = Dense(17, activation='sigmoid')(BN)
    
    model = Model(input=inputs, output=Dense2)
    
    return model


model = get_unet(channels, img_size, img_size)


epochs_arr  = [   20,      5,      5]
learn_rates = [0.001, 0.0003, 0.0001]

for learn_rate, epochs in zip(learn_rates, epochs_arr):
    if os.path.isfile(weights_path):
        print("loading existing weight for training")
        model.load_weights(weights_path)
    
    opt  = optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer=opt,
                  metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1),
                 ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=2)]

    model.fit(x = X_train, y= Y_train, validation_data=(X_val, Y_val),
          batch_size=256, verbose=2, epochs=epochs, callbacks=callbacks, shuffle=True)

if os.path.isfile(weights_path):
    model.load_weights(weights_path)


p_val = model.predict(X_val, batch_size = 128, verbose=1)
print(fbeta_score(Y_val, np.array(p_val) > 0.2, beta=2, average='samples'))

p_test = model.predict(x_test, batch_size = 128, verbose=1)





result = p_test
result = pd.DataFrame(result, columns = labels)

from tqdm import tqdm
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > 0.2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index))) 
    
df_test['tags'] = preds
df_test.to_csv('F:/DS-main/Kaggle-main/Planet Understanding the Amazon from Space/submission_unet.csv', index=False)

