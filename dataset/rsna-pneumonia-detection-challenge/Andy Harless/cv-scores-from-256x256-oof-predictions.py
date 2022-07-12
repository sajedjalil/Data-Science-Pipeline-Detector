import time
import numpy as np
import pandas as pd
import os
import gzip
import pickle
import csv
from skimage import measure
from skimage.transform import resize

print( os.listdir('../input/') )
compdir = '../input/rsna-pneumonia-detection-challenge/'
oofdir = '../input/oof-probabilities/'
print( os.listdir(compdir) )
print( os.listdir(oofdir) )
oofpath = oofdir + 'oof_probs.pkl.gz'
boxpath = compdir + 'stage_1_train_labels.csv'
imgdir = compdir + 'stage_1_train_images'


# Read OOF data

f = gzip.open(oofpath,'rb')
oofdict = pickle.load(f)
f.close()


# Load true opacity locations

# empty dictionary
pneumonia_locations = {}
# load table
with open(os.path.join(boxpath), mode='r') as infile:
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



# Evaluation metric functions, from Yicheng Chen's kernel:
#   https://www.kaggle.com/chenyc15/mean-average-precision-metric

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
    
    
    
# Calculate evaluation metrics for OOF predictions, at various probability thresholds

prob_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# prob_thresholds = [0.5]
nthresh = len(prob_thresholds)

count = 0
ns = nthresh*[0]
nfps = nthresh*[0]
ntps = nthresh*[0]
overall_maps = nthresh*[0.]
for patient in oofdict:
    # get predictions for this patient
    pred = oofdict[patient] / 255.
    count = count + 1
    # threshold predicted mask
    boxes_preds = []
    scoress = []
    # calculate predicted boxes and confidences at each probability threshold
    for thresh in prob_thresholds:
        comp = pred[:, :] > thresh
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
    # if image contains pneumonia
    if patient in pneumonia_locations:
        # loop through pneumonia
        for location in pneumonia_locations[patient]:
            x, y, w, h = [d//4 for d in location]
            boxes_true = np.append(boxes_true, [[x, y, w, h]], axis=0)
    # for each probability threshold, calculate evaluation metric
    for i in range(nthresh):
        if ( boxes_true.shape[0]==0 and boxes_preds[i].shape[0]>0 ):
            ns[i] = ns[i] + 1
            nfps[i] = nfps[i] + 1
        elif ( boxes_true.shape[0]>0 ):
            ns[i] = ns[i] + 1
            contrib = map_iou( boxes_true, boxes_preds[i], scoress[i] ) 
            overall_maps[i] = overall_maps[i] + contrib
            if ( boxes_pred.shape[0]>0 ):
                ntps[i] = ntps[i] + 1

for i, thresh in enumerate(prob_thresholds):
    print( "\nProbability threshold ", thresh )
    overall_maps[i] = overall_maps[i] / (ns[i]+1e-7)
    print( "False positive cases:  ", nfps[i] )
    print( "True positive cases: ", ntps[i] )
    print( "Overall evaluation score: ", overall_maps[i] )