# Many thanks to all the kagglers who contributed with bounding box annotations
# Also many thanks to kagglers who contributed with kernels and examples 
# This kernel is inspired by them and hopes to help and inspire others as well

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import time
import json
from PIL import Image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#from load import process_train, process_validation, process_test
from keras.applications import VGG16
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.utils import np_utils

nb_classes = 8
data_dir = '../input'
train_dir = 'train'
#valid_dir = 'valid'
test_dir = 'test_stg1'
bbox_dir = 'put/here/the/path/to/boundig_box/files'

np.random.seed(2017)


def read_img(path, shape):
    img = Image.open(path)
    img_size = img.size
    img = img.resize(shape, Image.ANTIALIAS)
    img = np.asarray(img)
    return img, img_size


# go through each image in each folder and read it
def load_train(shape):
    trX = []
    trX_id = []
    trY = []
    trX_img_sizes = dict()
    start = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for folder in folders:
        index = folders.index(folder)
        print('Load folder {} (Index: {})'.format(folder, index))
        path = os.path.join(data_dir, train_dir, folder, '*.jpg')
        files = glob.glob(path)
        for file_n in files:
            filename = os.path.basename(file_n)
            img, img_size = read_img(file_n, shape)
            trX.append(img)
            trX_id.append(filename)
            trY.append(index)
            trX_img_sizes[filename] = img_size

    end = time.time()
    print('Read train data time: {} seconds'
          .format(round(end - start, 2)))

    return trX, trY, trX_id, trX_img_sizes


# just transform data into numpy array
def process_train(shape):
    trX, trY, trX_id, trX_img_sizes = load_train(shape)

    print('Convert to  numpy...')
    trX = np.array(trX, dtype=np.uint8)
    trY = np.array(trY, dtype=np.uint8)

    print('Convert to float...')
    trX = trX.astype('float32')
   
    trY = np_utils.to_categorical(trY, nb_classes)

    print('Train shape: ', trX.shape)
    print(trX.shape[0], 'train samples')

    return trX, trY, trX_id, trX_img_sizes


# go through each image in the test folder and read it
def load_test(shape):
    path = os.path.join(data_dir, test_dir, '*.jpg')
    files = sorted(glob.glob(path))

    teX = []
    teX_id = []
    for file_n in files:
        file_name = os.path.basename(file_n)
        img, _ = read_img(file_n, shape)
        teX.append(img)
        teX_id.append(file_name)

    return teX, teX_id


# transform test data into numpy array
def process_test(shape):
    start = time.time()
    teX, teX_id = load_test(shape)

    teX = np.array(teX, dtype=np.uint8)

    teX = teX.astype('float32')
    # teX /= 255

    print('Test shape: ', teX.shape)
    print(teX.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'
          .format(round(time.time() - start, 2)))

    return teX, teX_id



def read_bbox(filenames):
    classes = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
    path = os.path.join(data_dir, bbox_dir)
    bb_json = {}
    for cls in classes:
        js = json.load(open('{}/{}.json'.format(path, cls), 'r'))
        for lbl in js:
            if 'annotations' in lbl.keys() and len(lbl['annotations']) > 0:
                bb_json[lbl['filename'].split('/')[-1]] = sorted(
                    lbl['annotations'], key=lambda x: x['height'] * x['width']
                )[-1]

    file2idx = {o: i for i, o in enumerate(filenames)}
    empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}

    for f in filenames:
        if f not in bb_json.keys():
            bb_json[f] = empty_bbox

    return bb_json


def convert_bbox(bbox, size):
    bbox_params = ['height', 'width', 'x', 'y']
    bbox = [bbox[p] for p in bbox_params]
    conv_x = (224. / size[0])
    conv_y = (224. / size[1])
    bbox[0] = bbox[0] * conv_y
    bbox[1] = bbox[1] * conv_x
    bbox[2] = max(bbox[2] * conv_x, 0)
    bbox[3] = max(bbox[3] * conv_y, 0)
    return bb


def process_bbox(filenames, sizes):
    bboxes = read_bbox(filenames)
    trX_bbox = np.stack([convert_bbox(bboxes[f], sizes[f])
                         for f in filenames],
                        ).astype(np.float32)


if __name__=="__main__":
    trX, trY, trX_id, trX_img_sizes = process_train((224, 224))
    print('\n')
    print('Target shape: ', trY.shape)
    
    valX = trX[:500]
    valY = trY[:500]
    valX_id = trX_id[:500]
    valX_img_sizes = {key: trX_img_sizes[key] for _, key in zip(range(500), trX_img_sizes)}
    
    print('\n')
    print('Validation shape: ', valX.shape)
    print(valX.shape[0], 'validation samples')
    
    trX_tmp = trX[500:]
    trY_tmp = trY[500:]
    trX_id_tmp = trX_id[500:]
    trX_img_sizes_tmp = {key: trX_img_sizes[key] for _, key in zip(range(500, len(trX_img_sizes)), trX_img_sizes)}
    
    print('\n')
    print('New train shape: ',trX_tmp.shape)
    print(trX_tmp.shape[0], 'new train samples')
    
    teX, teX_id = process_test((224, 224))
    
    # uncomment the below lines to get the bounding boxes
    # at the moment if you uncomment and run the code you'll get an error
    # since the bounding box files are missing from the directory
    #trX_bbox = process_bbox(trX_id_tmp, trX_img_sizes_tmp)
    #valX_bbox = process_bbox(valX, valX_img_sizes)
    
     # Classification block
    model = VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
    output = model.output
    output = Dropout(0.5)(output)
    output = Flatten()(output)
    output = Dense(4096, activation='relu')(output)
    output = Dropout(0.2)(output)
    output = Dense(4096, activation='relu')(output)
    output = Dense(nb_classes)(output)
    model = Model(model.input, output)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # change here trY_tmp with trX_bbox and valY with valX_bbox
    model.fit(trX_tmp, trY_tmp, batch_size=2, nb_epoch=25, shuffle=True, validation_data=(valX, valY), verbose=1)
    y_pred = model.predict(valX)