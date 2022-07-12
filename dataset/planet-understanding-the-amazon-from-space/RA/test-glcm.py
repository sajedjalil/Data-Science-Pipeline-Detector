# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data, color
from multiprocessing import Pool, cpu_count
from sklearn.metrics import fbeta_score
from skimage.measure import moments
from scipy import ndimage as ndi
from PIL import Image, ImageStat
from scipy import ndimage
from skimage import io
import glob, cv2
import random
import scipy
import cv2
import random
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor 
import xgboost as xgb
import numpy as np
import os, sys
import struct
import sys
import os
import subprocess
from six import string_types
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import random
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

def extract_features(df, data_path):
	features = df.copy()

	glcm_contrast_mean = []
	glcm_contrast_std = []
	glcm_dissimilarity_mean = []
	glcm_dissimilarity_std = []
	glcm_homogeneity_mean = []
	glcm_homogeneity_std = []
	glcm_correlation_mean = []
	glcm_correlation_std = []
	glcm_ASM_mean = []
	glcm_ASM_std = []
	glcm_energy_mean = []
	glcm_energy_std = []


	for image_name in tqdm(features.image_name.values):
		path = data_path + image_name + '.jpg'
		im = Image.open(path)
		r = np.array(im)[:,:,0]
		g = np.array(im)[:,:,1]
		b = np.array(im)[:,:,2]

		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		
		# GLCM features
		glcm = greycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
		contrast = greycoprops(glcm, 'contrast')
		dissimilarity = greycoprops(glcm, 'dissimilarity')
		homogeneity = greycoprops(glcm, 'homogeneity')
		correlation = greycoprops(glcm, 'correlation')
		ASM = greycoprops(glcm, 'ASM')
		energy = greycoprops(glcm, 'energy')
		
		glcm_contrast_mean.append(np.mean(contrast))
		glcm_contrast_std.append(np.std(contrast))
		glcm_dissimilarity_mean.append(np.mean(dissimilarity))
		glcm_dissimilarity_std.append(np.std(dissimilarity))		
		glcm_homogeneity_mean.append(np.mean(homogeneity))
		glcm_homogeneity_std.append(np.std(homogeneity))
		glcm_correlation_mean.append(np.mean(correlation))
		glcm_correlation_std.append(np.std(correlation))
		glcm_ASM_mean.append(np.mean(ASM))
		glcm_ASM_std.append(np.std(ASM))
		glcm_energy_mean.append(np.mean(energy))
		glcm_energy_std.append(np.std(energy))	

	features['glcm_contrast_mean'] = glcm_contrast_mean
	features['glcm_contrast_std'] = glcm_contrast_std
	features['glcm_dissimilarity_mean'] = glcm_dissimilarity_mean
	features['glcm_dissimilarity_std'] = glcm_dissimilarity_std
	features['glcm_homogeneity_mean'] = glcm_homogeneity_mean
	features['glcm_homogeneity_std'] = glcm_homogeneity_std
	features['glcm_correlation_mean'] = glcm_correlation_mean
	features['glcm_correlation_std'] = glcm_correlation_std
	features['glcm_ASM_mean'] = glcm_ASM_mean
	features['glcm_ASM_std'] = glcm_ASM_std
	features['glcm_energy_mean'] = glcm_energy_mean
	features['glcm_energy_std'] = glcm_energy_std
	
	return features
# Load data
train_path = '../input/train-jpg/'
test_path = '../input/test-jpg/'
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/sample_submission.csv')

# Extract features
print('Extracting train features')
train_features = extract_features(train, train_path)

print('Extracting test features')
test_features = extract_features(test, test_path)

# Prepare data
X = np.array(train_features.drop(['image_name', 'tags'], axis=1))
y_train = []

flatten = lambda l: [item for sublist in l for item in sublist]
labels = np.array(list(set(flatten([l.split(' ') for l in train_features['tags'].values]))))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for tags in tqdm(train.tags.values):
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    y_train.append(targets)
    
y = np.array(y_train, np.uint8)

print('X.shape = ' + str(X.shape))
print('y.shape = ' + str(y.shape))

n_classes = y.shape[1]

X_test = np.array(test_features.drop(['image_name', 'tags'], axis=1))

# Train and predict with one-vs-all strategy
y_pred = np.zeros((X_test.shape[0], n_classes))

print('Training and making predictions')
for class_i in tqdm(range(n_classes)): 
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, \
                              silent=True, objective='binary:logistic', nthread=-1, \
                              gamma=0, min_child_weight=1, max_delta_step=0, \
                              subsample=1, colsample_bytree=1, colsample_bylevel=1, \
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, \
                              base_score=0.5, seed=random_seed, missing=None)
    model.fit(X, y[:, class_i])
    y_pred[:, class_i] = model.predict_proba(X_test)[:, 1]

preds = [' '.join(labels[y_pred_row > 0.2]) for y_pred_row in y_pred]

subm = pd.DataFrame()
subm['image_name'] = test_features.image_name.values
subm['tags'] = preds
subm.to_csv('submission.csv', index=False)






