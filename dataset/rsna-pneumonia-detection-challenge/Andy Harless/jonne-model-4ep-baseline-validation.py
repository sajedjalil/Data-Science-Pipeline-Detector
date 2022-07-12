LR = 0.035
EPOCHS = 4
BATCHSIZE = 32
CHANNELS = 32
IMAGE_SIZE = 256
NBLOCK = 2 
DEPTH = 4
MOMENTUM = 0.9
METRIC_THRESH = 0.3  # Probability threshold to use for epoch-by-epoch IoU metric

DEBUG = False
DEBUG_SAMPLES = 2048

import os
import csv
import random
import pydicom
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt



# Load pneumonia locations

# empty dictionary
pneumonia_locations = {}
# load table
with open(os.path.join('../input/stage_1_train_labels.csv'), mode='r') as infile:
    # open reader
    reader = csv.reader(infile)
    # skip header
    next(reader, None)
    # loop through rows
    for rows in reader:
        # retrieve information
        filename = rows[0]
        location = rows[1:5]
        pneumonia = rows[5]
        # if row contains pneumonia add label to dictionary
        # which contains a list of pneumonia locations per filename
        if pneumonia == '1':
            # convert string to float to int
            location = [int(float(i)) for i in location]
            # save pneumonia location in dictionary
            if filename in pneumonia_locations:
                pneumonia_locations[filename].append(location)
            else:
                pneumonia_locations[filename] = [location]
                
                
# Load filenames

# load and shuffle filenames
folder = '../input/stage_1_train_images'
filenames = os.listdir(folder)
random.shuffle(filenames)
# split into train and validation filenames
n_valid_samples = 2560
if DEBUG:
    filenames = filenames[0:DEBUG_SAMPLES]
    n_valid_samples = DEBUG_SAMPLES//2
train_filenames = filenames[n_valid_samples:]
valid_filenames = filenames[:n_valid_samples]
print('n train samples', len(train_filenames))
print('n valid samples', len(valid_filenames))
n_train_samples = len(filenames) - n_valid_samples


# Data generator

class generator(keras.utils.Sequence):
    
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=BATCHSIZE, 
                 image_size=IMAGE_SIZE, shuffle=True, augment=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains pneumonia
        if filename in pneumonia_locations:
            # loop through pneumonia
            for location in pneumonia_locations[filename]:
                # add 1's at the location of the pneumonia
                x, y, w, h = location
                msk[y:y+h, x:x+w] = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)
            
            
# Network

def create_downsample(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=MOMENTUM)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x = keras.layers.MaxPool2D(2)(x)
    return x

def create_resblock(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=MOMENTUM)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=MOMENTUM)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return keras.layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=NBLOCK, depth=DEPTH):
    # input
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = keras.layers.BatchNormalization(momentum=MOMENTUM)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = keras.layers.UpSampling2D(2**depth)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
    


# IoU metric functions using tf.py_func, as suggested in  Marsh's (@vbookshelf) kernel:
#   https://www.kaggle.com/vbookshelf/keras-iou-metric-implemented-without-tensor-drama

def raw_iou(y_true, y_pred):
    results = []
    y_pred = y_pred > METRIC_THRESH
    for i in range(0,y_true.shape[0]):
        intersect = np.sum( y_true[i,:,:] * y_pred[i,:,:] )
        union = np.sum( y_true[i,:,:] ) + np.sum( y_pred[i,:,:] ) - intersect + 1e-7
        iou = np.mean((intersect/union)).astype(np.float32)
        results.append( iou )
    return np.mean( results )

def IoU(y_true, y_pred):
    iou = tf.py_func(raw_iou, [y_true, y_pred], tf.float32)
    return iou



# Train network

# create network and compiler
model = create_network(input_size=IMAGE_SIZE, channels=CHANNELS, n_blocks=NBLOCK, depth=DEPTH)
model.compile(optimizer=keras.optimizers.Adam(lr=LR),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy', IoU])

# create train and validation generators
folder = '../input/stage_1_train_images'
train_gen = generator(folder, train_filenames, pneumonia_locations, batch_size=BATCHSIZE, 
                      image_size=IMAGE_SIZE, shuffle=True, augment=True, predict=False)
valid_gen = generator(folder, valid_filenames, pneumonia_locations, batch_size=BATCHSIZE, 
                      image_size=IMAGE_SIZE, shuffle=False, predict=False)

if DEBUG:
    EPOCHS = 1

history = model.fit_generator(train_gen, validation_data=valid_gen, 
                              epochs=EPOCHS, shuffle=True, verbose=2)
                              

                              
# Evaluation metric functions, from Yicheng Chen's kernel

# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union
    
def map_iou(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)
        
    # return None if both are empty, don't count the image in final evaluation (?)
    
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    
    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    
    map_total = 0
    
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m
    
    return map_total / len(thresholds)
    
    
# Predict validation images

prob_thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
nthresh = len(prob_thresholds)

# load and shuffle filenames
folder = '../input/stage_1_train_images'
test_filenames = valid_filenames
print('n test samples:', len(test_filenames))

# create test generator with predict flag set to True
test_gen = generator(folder, test_filenames, None, batch_size=25, 
                     image_size=IMAGE_SIZE, shuffle=False, predict=True)

# loop through validation set
count = 0
ns = nthresh*[0]
nfps = nthresh*[0]
ntps = nthresh*[0]
overall_maps = nthresh*[0.]
for imgs, filenames in test_gen:
    # predict batch of images
    preds = model.predict(imgs)
    # loop through batch
    for pred, filename in zip(preds, filenames):
        count = count + 1
        maxpred = np.max(pred)
        # resize predicted mask
        pred = resize(pred, (1024, 1024), mode='reflect')
        # threshold predicted mask
        boxes_preds = []
        scoress = []
        for thresh in prob_thresholds:
            comp = pred[:, :, 0] > thresh
            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            boxes_pred = np.empty((0,4),int)
            scores = np.empty((0))
            for region in measure.regionprops(comp):
                # retrieve x, y, height and width
                y, x, y2, x2 = region.bbox
                boxes_pred = np.append(boxes_pred, [[x, y, x2-x, y2-y]], axis=0)
                # proxy for confidence score
                conf = np.mean(pred[y:y2, x:x2])
                scores = np.append( scores, conf )
            boxes_preds = boxes_preds + [boxes_pred]
            scoress = scoress + [scores]
        boxes_true = np.empty((0,4),int)
        fn = filename.split('.')[0]
        # if image contains pneumonia
        if fn in pneumonia_locations:
            # loop through pneumonia
            for location in pneumonia_locations[fn]:
                x, y, w, h = location
                boxes_true = np.append(boxes_true, [[x, y, w, h]], axis=0)
        for i in range(nthresh):
            if ( boxes_true.shape[0]==0 and boxes_preds[i].shape[0]>0 ):
                ns[i] = ns[i] + 1
                nfps[i] = nfps[i] + 1
            elif ( boxes_true.shape[0]>0 ):
                ns[i] = ns[i] + 1
                contrib = map_iou( boxes_true, boxes_preds[i], scoress[i] ) 
                overall_maps[i] = overall_maps[i] + contrib
                if ( boxes_preds[i].shape[0]>0 ):
                    ntps[i] = ntps[i] + 1

    # stop if we've got them all
    if count >= len(test_filenames):
        break

for i, thresh in enumerate(prob_thresholds):
    print( "\nProbability threshold ", thresh )
    overall_maps[i] = overall_maps[i] / (ns[i]+1e-7)
    print( "False positive cases:  ", nfps[i] )
    print( "True positive cases: ", ntps[i] )
    print( "Overall evaluation score: ", overall_maps[i] )