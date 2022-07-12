import numpy as np
import pandas as pd
import pickle
import skimage
from skimage import io, transform, exposure
from matplotlib import pyplot as plt
from skimage.transform import *
import os
from keras.layers import Input, Dense, MaxPooling2D, Conv2D, UpSampling2D, Reshape, Cropping2D, Flatten
from keras.models import *
from keras import losses

MAX_NUM = 20000

csv = pd.read_csv(r'../input/train_v2.csv')
ser = csv['tags'].map(lambda x : x.split())
tags = []
for s in ser:
    tags = tags + s

sel_tags = ['road', 'water', 'agriculture', 'habitation']
def is_within_labels(x):
    x = x.split()
    for i in x:
        if i in sel_tags:
            return True
    return False
sel_rows = csv['tags'].map(is_within_labels)
np.sum(sel_rows)

sel_csv = csv.loc[sel_rows]

ims = []
base_path = r'../input/train-jpg'
counts = 0
for index in sel_csv.index:
    im_path = os.path.join(base_path, sel_csv.ix[index]['image_name'] + '.jpg')
    ti = io.imread(im_path)
    ti = resize(ti, (50, 50), mode='edge')
    ims.append(ti)
    counts = counts + 1
    if(counts == MAX_NUM): break
ims = np.array(ims)
print(ims.shape)

data = ims

noise_factor = 0.2
data_noisy = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
data_noisy = np.clip(data_noisy, 0., 1.)
print(data_noisy.shape)

input_shape = data.shape[1:]
input_img = Input(shape=input_shape)

x = Conv2D(32, (2, 2), activation='relu', padding='same', data_format='channels_last')(input_img)
x = MaxPooling2D((2, 2), padding='same', data_format='channels_last')(x)
encoded = x

x = Conv2D(32, (2, 2), activation='relu', padding='same', data_format='channels_last')(encoded)
x = UpSampling2D((2, 2), data_format='channels_last')(x)
decoded = Conv2D(3, (1, 1), activation='sigmoid', padding='same', data_format='channels_last')(x)

autoencoder = Model(input_img, decoded)
opt = optimizers.Adadelta()
autoencoder.compile(optimizer=opt, loss=losses.binary_crossentropy)

autoencoder.fit(data_noisy, data, epochs=3, batch_size=20, shuffle=True, verbose=True)

denoised_data = autoencoder.predict(data)

n = 10
plt.figure(figsize=(40, 10))
for i in range(n):
    i = i + 1
    ax = plt.subplot(2, n, i)
    plt.imshow(data[i], aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i + n)
    plt.imshow(denoised_data[i], aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()