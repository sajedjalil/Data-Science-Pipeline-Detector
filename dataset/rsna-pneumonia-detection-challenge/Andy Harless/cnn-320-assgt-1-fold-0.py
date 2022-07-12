FOLD = 0
ASSGT = 1
ENV = 'Kaggle'
DEBUG = False

LR = .035
EPOCHS = 12
DECAY = 3e-5
BATCH_SIZE = 16
CHANNELS = 32
IMAGE_SIZE = 320
NBLOCK = 2
DEPTH = 4
MOMENTUM = 0.95
XFORM = True  # Transform input
CENTER = 0.5  # Center for input sigmoid transform
STRETCH = 8.0  # Width for input sigmoid  transform
STDIZE = True # Standardize input
STDMEAN = 0.48 # Standard mean
STDSD = 0.22 # Standard standard deviation
USE_ELLIPSE = False
TRAIN_MARGIN = 0
INFERENCE_MARGIN = 0
METRIC_THRESH = 0.3
THRESH = 0.3

DEBUG_SAMPLES = 2048
DEBUG_EPOCHS = 2

if ENV=='Kaggle':
    TRAIN_LABEL_FILE = '../input/make-folds/stage_1_train_labels_folds.csv'
    TRAIN_IMAGE_FOLDER = '../input/rsna-pneumonia-detection-challenge/stage_1_train_images'
    TEST_IMAGE_FOLDER ='../input/rsna-pneumonia-detection-challenge/stage_1_test_images'

if DEBUG:
    EPOCHS=DEBUG_EPOCHS

import os
import csv
import random
import pydicom
import gc
import gzip
import pickle
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize
from skimage.draw import ellipse
from scipy.special import expit

import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt






df = pd.read_csv(TRAIN_LABEL_FILE)


# Load pneumonia locations

# empty dictionary
pneumonia_locations = {}

for i, r in df.iterrows():
    if r.Target==1:
        filename = r.patientId
        x = int(r.x)
        y = int(r.y)
        w = int(r.width)
        h = int(r.height)
        location = [x,y,w,h]
        # if row contains pneumonia add label to dictionary
        # which contains a list of pneumonia locations per filename
        # save pneumonia location in dictionary
        if filename in pneumonia_locations:
            pneumonia_locations[filename].append(location)
        else:
            pneumonia_locations[filename] = [location]


# load and shuffle filenames
folder = TRAIN_IMAGE_FOLDER
# filenames = os.listdir(folder)
# split into train and validation filenames
if ASSGT==1:
    fold_id = df.fold
elif ASSGT==2:
    fold_id = df.fold2
else:
    print( 'Error: fold assignment does not exist' )
train_filenames = df[fold_id!=FOLD].patientId.unique() + '.dcm'
valid_filenames = df[fold_id==FOLD].patientId.unique() + '.dcm'
random.shuffle(train_filenames)
random.shuffle(valid_filenames)
if DEBUG:
    train_samples = DEBUG_SAMPLES//2
    valid_samples = DEBUG_SAMPLES//2
    test_samples = DEBUG_SAMPLES//2
    train_filenames = train_filenames[:train_samples]
    valid_filenames = valid_filenames[:valid_samples]
print('n train samples', len(train_filenames))
print('n valid samples', len(valid_filenames))


# Data generator

class generator(keras.utils.Sequence):
    
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=True, augment=False, predict=False):
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
                if USE_ELLIPSE:
                    rr, cc = ellipse( y+h/2, x+w/2, h/2+TRAIN_MARGIN, w/2+TRAIN_MARGIN, shape=img.shape )
                    msk[rr, cc] = 1
                else:
                    msk[y:y+h, x:x+w] = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        if STDIZE:
            sd = img.reshape(-1).std()
            avg = img.reshape(-1).mean()
            img = (((img-avg)/sd)*STDSD)+STDMEAN
        if XFORM:
            img = expit(STRETCH*(img-CENTER))
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        if STDIZE:
            sd = img.reshape(-1).std()
            avg = img.reshape(-1).mean()
            img = (((img-avg)/sd)*STDSD)+STDMEAN
        if XFORM:
            img = expit(STRETCH*(img-CENTER))
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

def create_network(input_size, channels, n_blocks=2, depth=4):
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
    
    
# Train network

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
    
# create network and compiler
model = create_network(input_size=IMAGE_SIZE, channels=CHANNELS, n_blocks=NBLOCK, depth=DEPTH)
model.compile(optimizer=keras.optimizers.Adam(lr=LR, decay=DECAY),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy', IoU])

# create train and validation generators
folder = TRAIN_IMAGE_FOLDER
train_gen = generator(folder, train_filenames, pneumonia_locations, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=True, augment=True, predict=False)
valid_gen = generator(folder, valid_filenames, pneumonia_locations, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=False, predict=False)

history = model.fit_generator(train_gen, validation_data=valid_gen, 
                              epochs=EPOCHS, shuffle=True, verbose=2)
                              
                              
                              
# Plot

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(history.epoch, history.history["loss"], label="Train loss")
plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(132)
plt.plot(history.epoch, history.history["acc"], label="Train accuracy")
plt.plot(history.epoch, history.history["val_acc"], label="Valid accuracy")
plt.legend()
plt.subplot(133)
plt.plot(history.epoch, history.history["IoU"], label="Train IoU")
plt.plot(history.epoch, history.history["val_IoU"], label="Valid IoU")
plt.legend()
plt.savefig('epochs.png')

                              
                              
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

prob_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
nthresh = len(prob_thresholds)

# load and shuffle filenames
folder = TRAIN_IMAGE_FOLDER
test_filenames = valid_filenames
print('n test samples:', len(test_filenames))

# create test generator with predict flag set to True
test_gen = generator(folder, test_filenames, None, batch_size=25, image_size=IMAGE_SIZE, shuffle=False, predict=True)

oof_dict = {}
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

        # Save OOF predictions
        pred = resize(pred, (512, 512), mode='reflect')
        intpred = (pred * 255.).clip(0,255).astype(np.uint8)
        filename = filename.split('.')[0]
        oof_dict[filename] = intpred.copy()

        # Calculate OOF scores
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
                y = max( 0, y-INFERENCE_MARGIN )
                x = max( 0, x-INFERENCE_MARGIN )
                y2 = min( 1024, y2+INFERENCE_MARGIN )
                x2 = min( 1024, x2+INFERENCE_MARGIN )
                boxes_pred = np.append(boxes_pred, [[x, y, x2-x, y2-y]], axis=0)
                # proxy for confidence score
                conf = np.mean(pred[y:y2, x:x2])
                scores = np.append( scores, conf )
            boxes_preds = boxes_preds + [boxes_pred]
            scoress = scoress + [scores]
        boxes_true = np.empty((0,4),int)
        # if image contains pneumonia
        if filename in pneumonia_locations:
            # loop through pneumonia
            for location in pneumonia_locations[filename]:
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

outfile_oof = 'oof_probs_a' + str(ASSGT) + '_f' + str(FOLD) + '.pkl.gz'
f = gzip.open(outfile_oof,'wb')
pickle.dump(oof_dict,f)
f.close()

score_dict = {}
for i, thresh in enumerate(prob_thresholds):
    print( "\nProbability threshold ", thresh )
    overall_maps[i] = overall_maps[i] / (ns[i]+1e-7)
    score_dict[thresh] = {'nfp': nfps[i], 'ntp': ntps[i], 'map':overall_maps[i]}
    print( "False positive cases:  ", nfps[i] )
    print( "True positive cases: ", ntps[i] )
    print( "Overall evaluation score: ", overall_maps[i] )
scores = pd.DataFrame.from_dict(score_dict,orient='index')
outfile_score = 'score320_a' + str(ASSGT) + '_f' + str(FOLD) + '.csv'
scores.to_csv(outfile_score)
    
del oof_dict
gc.collect()


# load filenames
folder = TEST_IMAGE_FOLDER
test_filenames = os.listdir(folder)
if DEBUG:
    test_filenames = test_filenames[:test_samples]
print('n test samples:', len(test_filenames))

# create test generator with predict flag set to True
test_gen = generator(folder, test_filenames, None, batch_size=25, image_size=IMAGE_SIZE, shuffle=False, predict=True)

# create submission dictionary
submission_dict = {}
prediction_dict = {}
# loop through testset
for imgs, filenames in test_gen:
    # predict batch of images
    preds = model.predict(imgs)
    # loop through batch
    for pred, filename in zip(preds, filenames):
        # save predictions
        pred = resize(pred, (512, 512), mode='reflect')
        intpred = (pred * 255.).clip(0,255).astype(np.uint8)
        filename = filename.split('.')[0]
        prediction_dict[filename] = intpred.copy()
        # resize predicted mask
        pred = resize(pred, (1024, 1024), mode='reflect')
        # threshold predicted mask
        comp = pred[:, :, 0] > THRESH
        # apply connected components
        comp = measure.label(comp)
        # apply bounding boxes
        predictionString = ''
        for region in measure.regionprops(comp):
            # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            y = max( 0, y-INFERENCE_MARGIN )
            x = max( 0, x-INFERENCE_MARGIN )
            y2 = min( 1024, y2+INFERENCE_MARGIN )
            x2 = min( 1024, x2+INFERENCE_MARGIN )
            height = y2 - y
            width = x2 - x
            # proxy for confidence score
            conf = np.mean(pred[y:y+height, x:x+width])
            # add to predictionString
            predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
        # add filename and predictionString to dictionary
        filename = filename.split('.')[0]
        submission_dict[filename] = predictionString
    # stop if we've got them all
    if len(submission_dict) >= len(test_filenames):
        break

# save submission file
sub = pd.DataFrame.from_dict(submission_dict,orient='index')
sub.index.names = ['patientId']
sub.columns = ['PredictionString']
outfile_sub = 'sub320_a' + str(ASSGT) + '_f' + str(FOLD) + '.csv'
sub.to_csv(outfile_sub)
del submission_dict, sub
gc.collect()

# save prediction file
outfile_test = 'test_probs_a' + str(ASSGT) + '_f' + str(FOLD) + '.pkl.gz'
f = gzip.open(outfile_test,'wb')
pickle.dump(prediction_dict,f)
f.close()


