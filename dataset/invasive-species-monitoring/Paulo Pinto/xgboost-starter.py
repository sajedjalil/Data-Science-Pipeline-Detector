# -*- coding: utf-8 -*-
# Based on
# https://www.kaggle.com/the1owl/fractals-of-nature-blend-0-90050

from PIL import Image, ImageStat
from tqdm import tqdm
from sklearn import model_selection
import xgboost as xgb
import pandas as pd
import numpy as np
import glob, cv2
import scipy
import random
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

random.seed(4)
np.random.seed(4)

def get_features(path):
    try:
        ft = []
        img = Image.open(path)
        im_stats_ = ImageStat.Stat(img)
        ft += im_stats_.sum
        ft += im_stats_.mean
        ft += im_stats_.rms
        ft += im_stats_.var
        ft += im_stats_.stddev
        img = np.array(img)[:,:,:3]
        ft += [scipy.stats.kurtosis(img[:,:,0].ravel())]
        ft += [scipy.stats.kurtosis(img[:,:,1].ravel())]
        ft += [scipy.stats.kurtosis(img[:,:,2].ravel())]
        ft += [scipy.stats.skew(img[:,:,0].ravel())]
        ft += [scipy.stats.skew(img[:,:,1].ravel())]
        ft += [scipy.stats.skew(img[:,:,2].ravel())]
        bw = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        ft += list(cv2.HuMoments(cv2.moments(bw)).flatten())
        ft += list(cv2.calcHist([bw],[0],None,[64],[0,256]).flatten()) #bw 
        ft += list(cv2.calcHist([img],[0],None,[64],[0,256]).flatten()) #r
        ft += list(cv2.calcHist([img],[1],None,[64],[0,256]).flatten()) #g
        ft += list(cv2.calcHist([img],[2],None,[64],[0,256]).flatten()) #b
        m, s = cv2.meanStdDev(img) #mean and standard deviation
        ft += list(m.ravel())
        ft += list(s.ravel())
        ft += [cv2.Laplacian(bw, cv2.CV_64F).var()] 
        ft += [cv2.Laplacian(img, cv2.CV_64F).var()]
        ft += [cv2.Sobel(bw,cv2.CV_64F,1,0,ksize=5).var()]
        ft += [cv2.Sobel(bw,cv2.CV_64F,0,1,ksize=5).var()]
        ft += [cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5).var()]
        ft += [cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5).var()]
    except:
        print(path)
    return ft

def load_img(paths):
    imf_d = {}
    for f in tqdm(paths, miniters=30):
        imf_d[f] = get_features(f)
    fdata = [imf_d[f] for f in paths]
    return fdata

print('Loading Train Data')
in_path = '../input/'
train = pd.read_csv(in_path + 'train_labels.csv')
train['path'] = train['name'].map(lambda x: in_path + 'train/' + str(x) + '.jpg')
xtrain = load_img(train['path']); print('train...')
pd.DataFrame.from_dict(xtrain).to_csv('xtrain1.csv', index=False)
xtrain = pd.read_csv('xtrain1.csv')

print('Loading Test Data')
test_jpg = glob.glob(in_path + 'test/*.jpg')
test = pd.DataFrame([[p.split('/')[3].replace('.jpg',''),p] for p in test_jpg])
test.columns = ['name','path']
xtest = load_img(test['path']); print('test...')
pd.DataFrame.from_dict(xtest).to_csv('xtest1.csv', index=False)
xtest = pd.read_csv('xtest1.csv')
               
xtrain = xtrain.values
xtest = xtest.values       
y = train['invasive'].values

print('xgb fitting ...')
xgb_test = pd.DataFrame(test[['name']], columns=['name'])
y_pred = np.zeros(xtest.shape[0])
xgtest = xgb.DMatrix(xtest)
score = 0
folds = 3 #10
kf = model_selection.StratifiedKFold(n_splits=folds, shuffle=False, random_state=4)

print('Training and making predictions')
for trn_index, val_index in kf.split(xtrain, y):
    
    xgtrain = xgb.DMatrix(xtrain[trn_index], label=y[trn_index])
    xgvalid = xgb.DMatrix(xtrain[val_index], label=y[val_index])
    
    params = {
        'eta': 0.05, #0.03
        'silent': 1,
        'verbose_eval': True,
        'verbose': False,
        'seed': 4
    }
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = "auc"
    params['min_child_weight'] = 15
    params['cosample_bytree'] = 0.8
    params['cosample_bylevel'] = 0.9
    params['max_depth'] = 4
    params['subsample'] = 0.9
    params['max_delta_step'] = 10
    params['gamma'] = 1
    params['alpha'] = 0
    params['lambda'] = 1
    #params['base_score'] =  0.63
    
    watchlist = [ (xgtrain,'train'), (xgvalid, 'valid') ]
    model = xgb.train(list(params.items()), xgtrain, 5000, watchlist, 
                      early_stopping_rounds=25, verbose_eval = 50)
    
    y_pred += model.predict(xgtest,ntree_limit=model.best_ntree_limit)
    score += model.best_score

y_pred /= folds
score /= folds
print('Mean AUC:',score)

now = datetime.datetime.now()
xgb_test['invasive'] = y_pred
xgb_test[['name','invasive']].to_csv('sub_xgb_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'_'+
                                     str(round(score,5))+'.csv', index=False)
