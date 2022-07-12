import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb

import scipy
from sklearn.metrics import fbeta_score

from PIL import Image

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

# Load data
train_path = '../input/train-jpg/'
test_path = '../input/test-jpg/'
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/sample_submission.csv')

def extract_features(df, data_path):
    im_features = df.copy()

    r_mean = []
    g_mean = []
    b_mean = []

    r_std = []
    g_std = []
    b_std = []

    r_max = []
    g_max = []
    b_max = []

    r_min = []
    g_min = []
    b_min = []

    r_kurtosis = []
    g_kurtosis = []
    b_kurtosis = []
    
    r_skewness = []
    g_skewness = []
    b_skewness = []

    for image_name in tqdm(im_features.image_name.values, miniters=100): 
        im = Image.open(data_path + image_name + '.jpg')
        im = np.array(im)[:,:,:3]

        r_mean.append(np.mean(im[:,:,0].ravel()))
        g_mean.append(np.mean(im[:,:,1].ravel()))
        b_mean.append(np.mean(im[:,:,2].ravel()))

        r_std.append(np.std(im[:,:,0].ravel()))
        g_std.append(np.std(im[:,:,1].ravel()))
        b_std.append(np.std(im[:,:,2].ravel()))

        r_max.append(np.max(im[:,:,0].ravel()))
        g_max.append(np.max(im[:,:,1].ravel()))
        b_max.append(np.max(im[:,:,2].ravel()))

        r_min.append(np.min(im[:,:,0].ravel()))
        g_min.append(np.min(im[:,:,1].ravel()))
        b_min.append(np.min(im[:,:,2].ravel()))

        r_kurtosis.append(scipy.stats.kurtosis(im[:,:,0].ravel()))
        g_kurtosis.append(scipy.stats.kurtosis(im[:,:,1].ravel()))
        b_kurtosis.append(scipy.stats.kurtosis(im[:,:,2].ravel()))
        
        r_skewness.append(scipy.stats.skew(im[:,:,0].ravel()))
        g_skewness.append(scipy.stats.skew(im[:,:,1].ravel()))
        b_skewness.append(scipy.stats.skew(im[:,:,2].ravel()))


    im_features['r_mean'] = r_mean
    im_features['g_mean'] = g_mean
    im_features['b_mean'] = b_mean

    im_features['r_std'] = r_std
    im_features['g_std'] = g_std
    im_features['b_std'] = b_std

    im_features['r_max'] = r_max
    im_features['g_max'] = g_max
    im_features['b_max'] = b_max

    im_features['r_min'] = r_min
    im_features['g_min'] = g_min
    im_features['b_min'] = b_min

    im_features['r_kurtosis'] = r_kurtosis
    im_features['g_kurtosis'] = g_kurtosis
    im_features['b_kurtosis'] = b_kurtosis
    
    im_features['r_skewness'] = r_skewness
    im_features['g_skewness'] = g_skewness
    im_features['b_skewness'] = b_skewness
    
    return im_features


