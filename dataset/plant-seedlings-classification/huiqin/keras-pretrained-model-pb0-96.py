#encoding=utf-8
'''
Based on https://www.kaggle.com/xingyuyang/cnn-with-keras

Since we used the InceptionResNetV2 pretrained model, hence the program will be slow running in cpu. 
If you want to speed up, you can change the img_size into smaller (e.g. img_size=100).

I run it on my local pc and get a 0.96 score in PB.
'''
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
np.random.seed(2)

from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer



import cv2
import glob
import random
import pandas as pd

from keras.models import *
from keras.optimizers import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.preprocessing import image
from keras.callbacks import *
from keras.metrics import top_k_categorical_accuracy
#============ Multi-GPU ==========
#from multi_gpu import to_multi_gpu

import cv2
img_size=299#48
img_shape=(img_size, img_size)
z = glob.glob('../input/train/*/*.png')
print("train size ",len(z))
ori_label = []
ori_imgs = []

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))


    return image


def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
    return image

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def get_img(path, img_rows, img_cols):

    img = image.load_img(path, target_size=(img_rows, img_cols))
    img = image.img_to_array(img)
    return img

for fn in z:
    if fn[-3:] != 'png':
        continue
    ori_label.append(fn.split('/')[-2])#unix '/',windows '\\'
    new_img=get_img(fn,img_size,img_size)
    ori_imgs.append(new_img)


imgs=[]
for im in ori_imgs:
    im=np.array(im)
    im=preprocess_input(im)
    im = randomHorizontalFlip(im)
    im=randomShiftScaleRotate(im)
    imgs.append(im)



imgs=np.array(imgs)
imgs = imgs.reshape(imgs.shape[0], img_size, img_size, 3)
from keras.applications import inception_resnet_v2


lb = LabelBinarizer().fit(ori_label)
label = lb.transform(ori_label)


trainX, validX, trainY, validY = train_test_split(imgs, label, test_size=0.1, random_state=2)
print(trainX.shape,validX.shape)

def incept_resnet(num_class, input_shape=(299, 299, 3)):
    base_model = InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(
            shape=input_shape))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_class, activation='softmax')(x)

    model = Model(input=base_model.input, output=predictions)



    # ============ Multi-GPU ============
    #model = to_multi_gpu(model, n_gpus=2)
    # ===================================

    from keras.optimizers import Adam
    optm=Adam(lr=0.0001)
    model.compile(optimizer=optm, loss='categorical_crossentropy',metrics=['acc'])
    return model


from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint


model=incept_resnet(12)
batch_size = 32#48
model_filepath='model3.h5'
callbacks = [EarlyStopping(monitor='val_loss',
                               patience=2,
                               verbose=1,
                               min_delta=1e-4,
                               mode='min'),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=1,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='min'),
                 ModelCheckpoint(monitor='val_loss',
                                 filepath=model_filepath,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min'),

                 ]
model.fit(
    trainX, trainY, batch_size=batch_size,
    epochs=20,
    verbose=2,
    validation_data=(validX, validY),
    callbacks=callbacks
)

z = glob.glob('../input/test/*.png')
test_imgs = []
names = []
for fn in z:
    if fn[-3:] != 'png':
        continue
    names.append(fn.split('/')[-1])#unix '/',win '\\'
    new_img = get_img(fn, img_size, img_size)
    test_imgs.append(new_img)

model.load_weights(model_filepath)

timgs = np.array([np.array(im) for im in test_imgs])
timgs=preprocess_input(timgs)
# testX = timgs.reshape(timgs.shape[0], img_size, img_size, 3) / 255
testX = timgs.reshape(timgs.shape[0], img_size, img_size, 3)


yhat = model.predict(testX)
test_y = lb.inverse_transform(yhat)

df = pd.DataFrame(data={'file': names, 'species': test_y})
df_sort = df.sort_values(by=['file'])
df_sort.to_csv('results3.csv', index=False)

'''

Epoch 8/20
107s - loss: 0.0157 - acc: 0.9970 - val_loss: 0.1212 - val_acc: 0.9642
Epoch 9/20
106s - loss: 0.0077 - acc: 0.9984 - val_loss: 0.1212 - val_acc: 0.9621
Epoch 10/20
107s - loss: 0.0040 - acc: 0.9991 - val_loss: 0.1186 - val_acc: 0.9621
Epoch 11/20
106s - loss: 0.0034 - acc: 0.9998 - val_loss: 0.1197 - val_acc: 0.9663
Epoch 12/20
107s - loss: 0.0026 - acc: 1.0000 - val_loss: 0.1163 - val_acc: 0.9663

'''