import cv2
import pandas as pd 
import numpy as np 
import os
from tqdm import tqdm

import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint



train_csv = pd.read_csv('../input/aptos2019-blindness-detection/train.csv', index_col=0)		# csv file with targets to all training images
test_csv = pd.read_csv('../input/aptos2019-blindness-detection/test.csv', index_col=0)			# csv file with targets to all testing images

X_test = []
testing_filenames = []

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# preprocessing and sampling all testing images from competition database
for img_file in tqdm(os.listdir('../input/aptos2019-blindness-detection/test_images')):
	filename = os.path.splitext(img_file)[0]
	testing_filenames.append(filename)
	img_path = os.path.join('../input/aptos2019-blindness-detection/test_images', img_file)

	# read img
	img = cv2.imread(img_path, 1)
	# threshold image, find contours and crop black edges to increase quality of image during resizing
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt = max(contours, key=cv2.contourArea)
	x, y, w, h = cv2.boundingRect(cnt)
	img = img[y:y+h, x:x+w]
	# resize image
	img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_AREA)
	# append images and labels to lists
	X_test.append(img)
	


# LOAD MODELS AND CREATE SUBMISSION FILE

#bin_model = load_model('../input/aptos2019models/CNN-pos-neg-binary-classification-step1.h5')
#cat_model = load_model('../input/aptos2019models/RESNET-stage-categorical-classification-step2/RESNET-stage-categorical-classification-step2')
ovr_model = load_model('../input/aptos2019models/RESNET-Overall-Classification-Step0.h5')

print(ovr_model.summary())

predictions = ovr_model.predict(np.array(X_test))
predictions = predictions.argmax(axis=1).astype('int').flatten()

print(predictions.shape)
print(predictions)

final_predictions = {x : int(predictions[i]) for i, x in enumerate(testing_filenames)}


sample_submission = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample_submission.id_code = final_predictions.keys()
sample_submission.diagnosis = final_predictions.values()

print(len(sample_submission))

sample_submission.to_csv("submission.csv", index=False)


