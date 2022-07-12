# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:23:37 2019

@author: HO Zhen Wai Olivier
"""

import matplotlib.pyplot as pyplot
os.listdir('./')

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import itertools
from sklearn.metrics import *

test_loc = '../input/X_test.csv'
xtrain_loc = '../input/X_train.csv'
ytrain_loc = '../input/y_train.csv'

pd.options.display.max_columns = 100

test_df = pd.read_csv(test_loc)
xtrain_df = pd.read_csv(xtrain_loc)
ytrain_df = pd.read_csv(ytrain_loc)


#one pass to get combined data
combined_df = pd.merge(xtrain_df, ytrain_df, on='series_id')
combined_df.surface = combined_df.surface.astype('category')
surface_type = combined_df.surface.cat.categories 
#carpet, concrete, fine_concrete, hard_tiles, hard_tiles_large_space, soft_pvc, solft_tiles, tiled, wood

g = 9.81
# try to work with summarized statistics 
#c'est une vitesse angulaire, l'unité est le radian
#acceleration in the direction
#plusieurs valeurs de frottements pour 

#groupbysur = combined_df.groupby('surface')
#groupbyid = combined_df.groupby('series_id')

import math
from math import cos,sin
def compute_Euler_angle(q0,q1,q2,q3):
    phi = math.atan(2*(q0*q1+q2*q3)/(1-2*(q1**2+q2**2)))
    theta = math.asin(2*(q0*q2-q3*q1))
    psi = math.atan(2*(q0*q3+q1*q2)/(1-2*(q2**2+q3**2)))
    return [phi,theta,psi]

def coordinate_conv_matrix(angles): #assume 3-2-1
    phi,theta,psi = angles[0],angles[1],angles[2]
    m1 = pd.Series([cos(theta)*cos(psi),-cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi), sin(theta)*sin(psi)+cos(phi)*sin(theta)*cos(psi)])
    m2 = pd.Series([cos(theta)*sin(psi), cos(phi)*cos(psi)+sin(phi)*sin(theta)*sin(psi), -sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi)])
    m3 = pd.Series([-sin(theta) , sin(phi)*cos(theta), cos(phi)*cos(theta)])
    return pd.DataFrame([m1,m2,m3])
def compute_acc(orientation, acceleration, order = [1,2,3]):
    angles = compute_Euler_angle(orientation[0],orientation[1],orientation[2],orientation[3])
    return coordinate_conv_matrix([angles[2],angles[1],angles[0]]).dot(acceleration)

#computing feature and stuff
def preprocess(data):
    x, y, z, w = data['orientation_X'].tolist(),data['orientation_Y'].tolist(),data['orientation_Z'].tolist(),data['orientation_W'].tolist()
    ax,ay,az = data['linear_acceleration_X'].tolist(),data['linear_acceleration_Y'].tolist(),data['linear_acceleration_Z'].tolist()
    axx, ayy, azz = [], [], []
    for i in range(len(x)):
        H = compute_acc([x[i],y[i],z[i],w[i]],[ax[i],ay[i],az[i]],[1,2,3])
        axx.append(H[0])
        ayy.append(H[1])
        azz.append(H[2])
    
    data['in_acc_X'],data['in_acc_Y'],data['in_acc_Z'] = axx, ayy, azz
    return data

xtrain_df = preprocess(xtrain_df)
test_df = preprocess(test_df)
def fe(data):
    df = pd.DataFrame()
    data['sangularv'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 +data['angular_velocity_Z']**2)** 0.5
    data['slinacc'] = (data['in_acc_X']**2+ data['in_acc_Y']**2+data['in_acc_Z']**2)**0.5
    data['acc_vs_vel'] = data['slinacc']/data['sangularv']
     
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number','orientation_X','orientation_Y','orientation_Z','orientation_W']:
            continue
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2
    return df

X_train = fe(xtrain_df)
X_test = fe(test_df)

le = LabelEncoder()
ytrain_df['surface'] = le.fit_transform(ytrain_df['surface'])

def k_folds(X, y, X_test, k):
    folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=2019)
    y_test = np.zeros((X_test.shape[0], 9))
    y_oof = np.zeros((X.shape[0]))
    score = 0
    for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):
        clf =  RandomForestClassifier(n_estimators = 500, n_jobs = -1)
        clf.fit(X_train.iloc[train_idx], y[train_idx])
        y_oof[val_idx] = clf.predict(X.iloc[val_idx])
        y_test += clf.predict_proba(X_test) / folds.n_splits
        score += clf.score(X.iloc[val_idx], y[val_idx])
        print('Fold: {} score: {}'.format(i,clf.score(X.iloc[val_idx], y[val_idx])))
    print('Avg Accuracy', score / folds.n_splits) 
        
    return y_oof, y_test 

y_oof, y_test = k_folds(X_train, ytrain_df['surface'], X_test, k= 50)




confusion_matrix(y_oof,ytrain_df['surface'])

y_test = np.argmax(y_test, axis=1)
submission = pd.read_csv(os.path.join("../input/",'sample_submission.csv'))
submission['surface'] = le.inverse_transform(y_test)
submission.to_csv('submission.csv', index=False)
submission.head(10)
