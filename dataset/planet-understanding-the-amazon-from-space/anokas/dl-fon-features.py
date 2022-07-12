import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
import time
import pickle
from PIL import Image, ImageStat
from skimage import io
import cv2

import scipy
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
from multiprocessing import Pool

from PIL import Image

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

# Load data
train_path = '../input/train-jpg/'
test_path = '../input/test-jpg/'
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/sample_submission.csv')

def extract_features_thread(path):
	if '.jpg' not in path:
		path += '.jpg'
	st = []
	# pillow jpg
	img = Image.open(path)
	im_stats_ = ImageStat.Stat(img)
	st += im_stats_.sum
	st += im_stats_.mean
	st += im_stats_.rms
	st += im_stats_.var
	st += im_stats_.stddev
	img = np.array(img)[:, :, :3]
	st += [scipy.stats.kurtosis(img[:, :, 0].ravel())]
	st += [scipy.stats.kurtosis(img[:, :, 1].ravel())]
	st += [scipy.stats.kurtosis(img[:, :, 2].ravel())]
	st += [scipy.stats.skew(img[:, :, 0].ravel())]
	st += [scipy.stats.skew(img[:, :, 1].ravel())]
	st += [scipy.stats.skew(img[:, :, 2].ravel())]
	# cv2 jpg
	img = cv2.imread(path)
	bw = cv2.imread(path, 0)
	st += list(cv2.calcHist([bw], [0], None, [256], [0, 256]).flatten())  # bw
	st += list(cv2.calcHist([img], [0], None, [256], [0, 256]).flatten())  # r
	st += list(cv2.calcHist([img], [1], None, [256], [0, 256]).flatten())  # g
	st += list(cv2.calcHist([img], [2], None, [256], [0, 256]).flatten())  # b
	try:
		# skimage tif
		imgr = io.imread(path.replace('jpg', 'tif'))
		tf = imgr[:, :, 3]
		st += list(cv2.calcHist([tf], [0], None, [256], [0, 65536]).flatten())  # near ifrared
		ndvi = ((imgr[:, :, 3] - imgr[:, :, 0]) / (
		imgr[:, :, 3] + imgr[:, :, 0]))  # water ~ -1.0, barren area ~ 0.0, shrub/grass ~ 0.2-0.4, forest ~ 1.0
		st += list(np.histogram(ndvi, bins=20, range=(-1, 1))[0])
		ndvi = ((imgr[:, :, 3] - imgr[:, :, 1]) / (imgr[:, :, 3] + imgr[:, :, 1]))
		st += list(np.histogram(ndvi, bins=20, range=(-1, 1))[0])
		ndvi = ((imgr[:, :, 3] - imgr[:, :, 2]) / (imgr[:, :, 3] + imgr[:, :, 2]))
		st += list(np.histogram(ndvi, bins=20, range=(-1, 1))[0])
	except:
		st += [-1 for i in range(256)]
		st += [-2 for i in range(60)]
		print('err', path.replace('jpg', 'tif'))
	m, s = cv2.meanStdDev(img)  # mean and standard deviation
	st += list(m)
	st += list(s)
	st += [cv2.Laplacian(bw, cv2.CV_64F).var()]
	st += [cv2.Laplacian(img, cv2.CV_64F).var()]
	st += [cv2.Sobel(bw, cv2.CV_64F, 1, 0, ksize=5).var()]
	st += [cv2.Sobel(bw, cv2.CV_64F, 0, 1, ksize=5).var()]
	st += [cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5).var()]
	st += [cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5).var()]
	st += [(bw < 30).sum()]
	st += [(bw > 225).sum()]

	return st

def extract_features_multi(files, data_path):
	paths = [data_path + image_name for image_name in files]
	pool = Pool(12)
	X = pool.map(extract_features_thread, paths)
	print('Forming array')
	X = np.array(X)
	return X


# Extract features
print('Extracting train features')
t0 = time.time()
train = train.image_name.values
X = extract_features_multi(train, train_path)
print(time.time() - t0)

pickle.dump(X, open('features.bin', 'wb'), protocol=4)
print('Cached features')
