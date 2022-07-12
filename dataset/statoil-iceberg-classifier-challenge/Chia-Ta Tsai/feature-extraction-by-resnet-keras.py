#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cttsai
To extract features by ResNet50 then can be further trained by XGBoost or others
original idea forked and refactored from
https://www.kaggle.com/kelexu/pretrained-resnet-feature-xgb
"""
from tqdm import tqdm
#
import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt
#import for image processing
import cv2
from keras.applications import ResNet50
#evaluation
from sklearn.model_selection import train_test_split
import xgboost as xgb
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
    
    bands = np.stack((band1, band2,  0.5 * (band1 + band2)), axis=-1)
    del band1, band2
    
    return df, bands


def process(df, bands, output='tmp'):

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
    

###############################################################################
if __name__ == '__main__':
    
    np.random.seed(1017)
    target = 'is_iceberg'
    
    #Load data
    train, train_bands = read_jason(file='train.json', loc='../input/')
    test, test_bands = read_jason(file='test.json', loc='../input/')

    train_X = process(df=train, bands=train_bands, output='train')
    train_y = train[target].values

    test_X = process(df=test, bands=test_bands, output='test')

    #training
    print('evaluating performance...')
    split_seed = 25
        
    tmp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")    
    x1, x2, y1, y2 = train_test_split(train_X, train_y, test_size=0.1, random_state=split_seed)
    print('splitted: {0}, {1}'.format(x1.shape, x2.shape), flush=True)

    #XGB
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'seed': 99, 'silent': True}
    params['eta'] = 0.05
    params['max_depth'] = 4
    params['subsample'] = 0.9
    params['eval_metric'] = 'logloss'
    params['colsample_bytree'] = 0.7
    params['colsample_bylevel'] = 0.7
    params['max_delta_step'] = 1
    #params['gamma'] = 1
    #params['labmda'] = 1
    params['scale_pos_weight'] = 1.0
    params['seed'] = split_seed + 1
    nr_round = 2000
    min_round = 100
            
    model1 = xgb.train(params, 
                       xgb.DMatrix(x1, y1), 
                       nr_round,  
                       watchlist, 
                       verbose_eval=50, 
                       early_stopping_rounds=min_round)
        
    pred_xgb = model1.predict(xgb.DMatrix(test_X), ntree_limit=model1.best_ntree_limit+45)
    
    #
    subm = pd.DataFrame({'id': test['id'].values, target: pred_xgb})
    file = 'subm_{}_xgb.csv'.format(tmp)
    subm.to_csv(file, index=False, float_format='%.6f')
  

