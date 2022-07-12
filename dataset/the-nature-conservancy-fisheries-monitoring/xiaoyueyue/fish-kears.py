
import numpy as np
np.random.seed(2017)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version

def get_img_cv2(path):
	img_src = cv2.imread(path)
	img_resize = cv2.resize(img_src,(48,48),cv2.INTER_LINEAR)
	return img_resize
	
def load_train():
	x_train = []
	x_train_id = []
	y_train = []
	start_time = time.time()
	
	print("read train image...")
	folders = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
	
	for folder in folders:
		index = folders.index(folder)
		print("Load folder {}(Index: {})".format(folder,index))
		path = os.path.join('..','input','train',folder,'*.jpg')
		files = glob.glob(path)
		for file in files:
			flbase = os.path.basename(file)
			img_src = get_img_cv2(file)
			x_train.append(img_src)
			x_train_id.append(flbase)
			y_train.append(index)
	print ('Read train data time: {} seconds'.format(round(time.time()-start_time,2)))
	return x_train, y_train	, x_train_id
	
def read_and_normalize_train_data():
	train_data,train_target,train_id = load_train()
	
	print('convert to numpy...')
	train_data = np.array(train_data,dtype = np.uint8)
	train_target = np.array(train_target, dtype=np.uint8)
	print(train_data.shape)
	print ('Reshape...')
	train_data = train_data.transpose((0,3,1,2))	
	print(train_data.shape)
read_and_normalize_train_data()
