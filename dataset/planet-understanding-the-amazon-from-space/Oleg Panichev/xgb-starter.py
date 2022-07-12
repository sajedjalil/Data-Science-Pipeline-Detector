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
test_path = '../input/test-jpg-v2/'
train = pd.read_csv('../input/train_v2.csv')
test = pd.read_csv('../input/sample_submission_v2.csv')

def extract_features(df, data_path):
    im_features = df.copy()

    N = len(im_features.image_name.values)

    r_mean = np.zeros(N)
    g_mean = np.zeros(N)
    b_mean = np.zeros(N)

    r_std = np.zeros(N)
    g_std = np.zeros(N)
    b_std = np.zeros(N)

    r_max = np.zeros(N)
    g_max = np.zeros(N)
    b_max = np.zeros(N)

    r_min = np.zeros(N)
    g_min = np.zeros(N)
    b_min = np.zeros(N)

    r_kurtosis = np.zeros(N)
    g_kurtosis = np.zeros(N)
    b_kurtosis = np.zeros(N)
    
    r_skewness = np.zeros(N)
    g_skewness = np.zeros(N)
    b_skewness = np.zeros(N)

    for i, image_name in enumerate(tqdm(im_features.image_name.values, miniters=1000)): 
        im = Image.open(data_path + image_name + '.jpg')
        im = np.array(im)[:,:,:3]

        r = im[:,:,0].ravel()
        g = im[:,:,1].ravel()
        b = im[:,:,2].ravel()
        
        r_mean[i] = np.mean(r)
        g_mean[i] = np.mean(g)
        b_mean[i] = np.mean(b)

        r_std[i] = np.std(r)
        g_std[i] = np.std(g)
        b_std[i] = np.std(b)

        r_max[i] = np.max(r)
        g_max[i] = np.max(g)
        b_max[i] = np.max(b)

        r_min[i] = np.min(r)
        g_min[i] = np.min(g)
        b_min[i] = np.min(b)

        r_kurtosis[i] = scipy.stats.kurtosis(r)
        g_kurtosis[i] = scipy.stats.kurtosis(g)
        b_kurtosis[i] = scipy.stats.kurtosis(b)
        
        r_skewness[i] = scipy.stats.skew(r)
        g_skewness[i] = scipy.stats.skew(g)
        b_skewness[i] = scipy.stats.skew(b)


    im_features['r_mean'] = r_mean
    im_features['g_mean'] = g_mean
    im_features['b_mean'] = b_mean

    im_features['rgb_mean_mean'] = (r_mean + g_mean + b_mean)/3.0

    im_features['r_std'] = r_std
    im_features['g_std'] = g_std
    im_features['b_std'] = b_std

    im_features['rgb_mean_std'] = (r_std + g_std + b_std)/3.0

    im_features['r_max'] = r_max
    im_features['g_max'] = g_max
    im_features['b_max'] = b_max

    im_features['rgb_mean_max'] = (r_max + r_max + b_max)/3.0

    im_features['r_min'] = r_min
    im_features['g_min'] = g_min
    im_features['b_min'] = b_min

    im_features['rgb_mean_min'] = (r_min + g_min + b_min)/3.0

    im_features['r_range'] = r_max - r_min
    im_features['g_range'] = g_max - g_min
    im_features['b_range'] = b_max - b_min

    im_features['r_kurtosis'] = r_kurtosis
    im_features['g_kurtosis'] = g_kurtosis
    im_features['b_kurtosis'] = b_kurtosis
    
    im_features['r_skewness'] = r_skewness
    im_features['g_skewness'] = g_skewness
    im_features['b_skewness'] = b_skewness
    
    return im_features

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

for tags in tqdm(train.tags.values, miniters=1000):
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
for class_i in tqdm(range(n_classes), miniters=1): 
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, \
                              silent=True, objective='binary:logistic', nthread=-1, \
                              gamma=0, min_child_weight=1, max_delta_step=0, \
                              subsample=1, colsample_bytree=1, colsample_bylevel=1, \
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, \
                              base_score=0.5, seed=random_seed, missing=None)
    model.fit(X, y[:, class_i])
    y_pred[:, class_i] = model.predict_proba(X_test)[:, 1]

preds = [' '.join(labels[y_pred_row > 0.21]) for y_pred_row in y_pred]

subm = pd.DataFrame()
subm['image_name'] = test_features.image_name.values
subm['tags'] = preds
subm.to_csv('submission.csv', index=False)
