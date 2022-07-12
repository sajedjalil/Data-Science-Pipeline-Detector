#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: qufu
Add cv pipeline To extract features by ResNet50 then can be further trained by XGBoost or others
original idea forked and refactored from
cttsai  https://www.kaggle.com/cttsai/feature-extraction-by-resnet-keras
"""
MAX_XGB_ROUNDS = 2175
OPTIMIZE_XGB_ROUNDS = True
XGB_LEARNING_RATE = 0.03
XGB_EARLY_STOPPING_ROUNDS = 100
from xgboost import XGBClassifier
from tqdm import tqdm
#
import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt
#import for image processing
import cv2
import os
from keras.applications import ResNet50
#evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

#

###############################################################################
def read_jason(file, loc='../input/'):

    df = pd.read_json('{}{}'.format(loc, file))
    #df = df[:100]
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    #print(df['inc_angle'].value_counts())
    
    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)
    ida = df['id']
    bands = np.stack((band1, band2,  0.5 * (band1 + band2)), axis=-1)
    del band1, band2
    
    return df, bands, ida


def process(df, bands, output='tmp'):

    if os.path.exists('{}ResNet.csv'.format(output)):
        X = pd.read_csv('{}ResNet.csv'.format(output))
    else:
        w, h = 197, 197
        model = ResNet50(include_top=False,
                         weights='imagenet',
                         input_shape=(h, w, 3),
                         pooling='avg') # or 'max' min= 139

        bands = 0.5 + bands / 100.
        X = []

        for i in tqdm(bands, miniters=100):

            x = cv2.resize(i, (w, h)).astype(np.float32)
            x = np.expand_dims(x, axis=0)

            preds = model.predict(x, verbose=2)
            features_reduce = preds.squeeze()
            X.append(features_reduce)

        X = np.array(X)

        feats = ['f{:04d}'.format(f+1) for f in range(X.shape[1])]
        transResNet = pd.DataFrame(X, columns=feats)
        transResNet['id'] = df['id'].values
        transResNet.to_csv('{}ResNet.csv'.format(output), index=False)

        X = np.concatenate([X, df['inc_angle'].values[:, np.newaxis]], axis=-1)

    return X

    
#Load data
train, train_bands, id_train = read_jason(file='train.json', loc='../input/')
test, test_bands, id_test = read_jason(file='test.json', loc='../input/')

X = process(df=train, bands=train_bands, output='train')
y = train['is_iceberg'].values

test_df = process(df=test, bands=test_bands,output='test')

y_valid_pred = 0.0*y
y_test_pred = 0.0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1108, shuffle = True)
np.random.seed(1108)

# Set up classifier
xgbmodel = XGBClassifier(
                        n_estimators=MAX_XGB_ROUNDS,
                        max_depth=3,
                        objective="binary:logistic",
                        learning_rate=XGB_LEARNING_RATE,
                        subsample=.9,
                        min_child_weight=.2,
                        colsample_bytree=.9,
                        scale_pos_weight=1,
                        gamma=0,
                        reg_alpha=0.2,
                        reg_lambda=0.4,
                     )

# Run CV
for i, (train_index, test_index) in enumerate(kf.split(X)):

    # Create data for this fold
    y_train, y_valid = y[train_index], y[test_index]
    X_train, X_valid = X[train_index,:], X[test_index,:]
    X_test = test_df
    print( "\nFold ", i)

    # Run model for this fold
    if OPTIMIZE_XGB_ROUNDS:
        eval_set=[(X_valid,y_valid)]
        fit_model = xgbmodel.fit( X_train, y_train,
                               eval_set=eval_set,
                               eval_metric='logloss',
                               early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                               verbose=50
                             )
        print( "  Best N trees = ", xgbmodel.best_ntree_limit )
        print( "  Best gini = ", xgbmodel.best_score )
    else:
        fit_model = xgbmodel.fit( X_train, y_train )

    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid, ntree_limit=xgbmodel.best_ntree_limit+1)[:,1]
    print( "  Gini = ", log_loss(y_valid, pred) )
    y_valid_pred[test_index] = pred

    # Accumulate test set predictions
    probs = fit_model.predict_proba(X_test, ntree_limit=xgbmodel.best_ntree_limit+1)[:,1]
    y_test_pred += probs

    del X_test, X_train, X_valid, y_train

y_test_pred /= K  # Average test set predictions


print( "\nGini for full training set:" ,log_loss(y, y_valid_pred))

# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['is_iceberg'] = y_valid_pred
val.to_csv('resfeat+xgb_valid.csv', float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['is_iceberg'] = y_test_pred
sub.to_csv('resfeat+xgb_submit.csv', float_format='%.6f', index=False)