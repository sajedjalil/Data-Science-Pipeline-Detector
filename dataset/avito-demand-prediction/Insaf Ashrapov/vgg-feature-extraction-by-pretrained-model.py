import numpy as np
import pandas as pd
import time, os, glob
import cv2

from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD, Adam
from keras.layers import GlobalAveragePooling2D
from keras import Model
from keras.applications.imagenet_utils import preprocess_input

def get_model(num_class, input_size, feature_layer):
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=[input_size,input_size,3], classes=num_class)
    x = base_model.get_layer(feature_layer).output
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=base_model.input, outputs=x)
    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def test_generator(x_train, batch_size, input_size, shuffle=False):
    batch_index = 0
    n = x_train.shape[0]
    while 1:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)

        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0

        batch_x = []
        batch_id = index_array[current_index: current_index + current_batch_size]
        for id in batch_id:
            img = cv2.imread(x_train['path'][id]).astype(np.float32)
            img = img[:,:,::-1]
            img = preprocess_input(img)
            batch_x.append(img)
        batch_x = np.array(batch_x, np.float32)

        yield batch_x



feature_layer = "block5_conv3"

index_path = "input/index/"
index_list = sorted(glob.glob(index_path + "*")) # 1091756
len_index = len(index_list)
query_path = "input/query/"
index_list += sorted(glob.glob(query_path + "*")) # 114943
index_list = pd.DataFrame(index_list, columns=['path'])
input_size = 224

model = get_model(1,input_size, feature_layer)

batch_size = 224
gen_test = test_generator(index_list, batch_size, input_size)
feature = model.predict_generator(generator=gen_test,
                                     steps=np.ceil(index_list.shape[0] / batch_size),
                                     verbose=1)

np.save("output/vgg_feature_{}_index.npy".format(feature_layer), feature[:len_index])
np.save("output/vgg_feature_{}_query.npy".format(feature_layer), feature[len_index:])