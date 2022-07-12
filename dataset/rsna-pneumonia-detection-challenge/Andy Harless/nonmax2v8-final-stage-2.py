OTHRESH=0.01
PTHRESH1=0.5
PTHRESH2=0.5
PWEIGHT1=1.0
PWEIGHT2=1.0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

print(os.listdir("../input/mrcnn-inference-from-saved-wts-2dig-95-stage-2"))
print(os.listdir("../input/yolov3-inference-from-multiple-saved-wts-stage-2"))

df1 = pd.read_csv("../input/mrcnn-inference-from-saved-wts-2dig-95-stage-2/submission_mrcnn_higher.csv")
df2 = pd.read_csv("../input/yolov3-inference-from-multiple-saved-wts-stage-2/submission_yolo15300.csv")

# Implementation of non-max suppression from
#   https://github.com/jrosebr1/imutils/blob/master/imutils/object_detection.py
def non_max_suppression(boxes, probs=None, overlapThresh=OTHRESH):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = area

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int"), list(np.array(probs)[pick])

print( df1.head() )
print( df2.head() )

positivePatients1 = df1[~df1.PredictionString.isnull()]
positivePatients2 = df2[~df2.PredictionString.isnull()]
positiveBoth = pd.merge( positivePatients1, positivePatients2, on='patientId', how='inner').patientId
allPatients = list( pd.merge( df1, df2, on='patientId', how='outer').patientId.values )

box_dict = {}
conf_dict = {}

for i, r in positivePatients1.iterrows():
    s = r.PredictionString.split(' ')
    if s[-1]=='':
        s.pop()  # remove terminating null
    if s[0]=='':
        s.pop(0)  # remove initial null
    if ( len(s)%5 ):
        print( 'Bad prediction string.')
    boxes = []
    confs = []
    while len(s):
        conf = float(s.pop(0))
        x = int(round(float(s.pop(0))))
        y = int(round(float(s.pop(0))))
        w = int(round(float(s.pop(0))))
        h = int(round(float(s.pop(0))))
        if conf>PTHRESH1:
            boxes.append( [x,y,x+w,y+h] )
            confs.append( min(1.0, PWEIGHT1*conf) )
    if len(boxes):
        box_dict[r.patientId] = boxes
        conf_dict[r.patientId] = confs

for i, r in positivePatients2.iterrows():
    s = r.PredictionString.split(' ')
    if s[-1]=='':
        s.pop()  # remove terminating null
    if s[0]=='':
        s.pop(0)  # remove initial null
    if ( len(s)%5 ):
        print( 'Bad prediction string.')
    boxes = []
    confs = []
    if r.patientId in box_dict:
        boxes = box_dict[r.patientId]
        confs = conf_dict[r.patientId]
    while len(s):
        conf = float(s.pop(0))
        x = int(round(float(s.pop(0))))
        y = int(round(float(s.pop(0))))
        w = int(round(float(s.pop(0))))
        h = int(round(float(s.pop(0))))
        if conf>PTHRESH2:
            boxes.append( [x,y,x+w,y+h] )
            confs.append( min(1.0, PWEIGHT2*conf**0.5) ) ## some kind of polinomial normalization
    if len(boxes):
        box_dict[r.patientId] = boxes
        conf_dict[r.patientId] = confs
    
for i, k in enumerate(box_dict):
    if i<5:
        print(k, box_dict[k])
    if k not in conf_dict:
        print( 'Patient ' + k + ' missing from conf_dict' )

for i, k in enumerate(conf_dict):
    if i<5:
        print(k, conf_dict[k])
    if k not in box_dict:
        print( 'Patient ' + k + ' missing from box_dict' )

box_dict_nms = {}
conf_dict_nms = {}
for p in box_dict:
    boxes, confs = non_max_suppression(np.array(box_dict[p]), np.array(conf_dict[p]))
    box_dict_nms[p] = boxes
    conf_dict_nms[p] = confs
    
for i, k in enumerate(box_dict_nms):
    if i<5:
        print('Input '+ k + ':')
        print(conf_dict[k], box_dict[k])
        print('Output: ')
        print(conf_dict_nms[k], box_dict_nms[k])
    if k not in conf_dict_nms:
        print( 'Patient ' + k + ' missing from conf_dict_nms' )

sub_dict = {}
for p in allPatients:
    predictionString = ''
    if p in box_dict_nms:
        for box, conf in zip(box_dict_nms[p], conf_dict_nms[p]):
            # retrieve x, y, height and width
            x, y, x2, y2 = box
            height = y2 - y
            width = x2 - x
            # add to predictionString
            predictionString += f'{conf:.02} {x} {y} {width} {height} '
    if len(predictionString) == 0:
        predictionString = None
    sub_dict[p] = predictionString

for i, k in enumerate(sub_dict):
    if i<5:
        print(k, sub_dict[k])

# save submission file
sub = pd.DataFrame.from_dict(sub_dict,orient='index')
sub.index.names = ['patientId']
sub.columns = ['PredictionString']
outfile_sub = 'sub_nms.csv'
sub.to_csv(outfile_sub)
