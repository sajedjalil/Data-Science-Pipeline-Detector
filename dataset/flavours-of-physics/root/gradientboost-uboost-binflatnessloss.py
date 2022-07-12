# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:50:40 2015

@author: ankit chaudhary
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 22:20:05 2015

@author: sony
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


from hep_ml.gradientboosting import UGradientBoostingClassifier,BinFlatnessLossFunction




train = pd.read_csv('../input/training.csv')

#randomize the training sample
train=train.iloc[np.random.permutation(len(train))]

test = pd.read_csv('../input/test.csv')

print("Eliminate SPDhits, which makes the agreement check fail")
features= ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP',
       'IPSig', 'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree',
       'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc',
       'isolationd', 'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2',
       
       
       'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_IP',
       'p1_IP', 'p2_IP', 'p0_IPSig', 'p1_IPSig', 'p2_IPSig', 
            'p0_eta', 'p1_eta',
       'p2_eta','mass']

print("Train a Random Fores and gradient boos model model")



print("train a UBoost classifier")
loss_funct=BinFlatnessLossFunction(uniform_features=['mass'],n_bins=5,uniform_label=0)
ub=UGradientBoostingClassifier(loss=loss_funct,n_estimators=100, random_state=3,learning_rate=0.25,subsample=0.7)
ub.fit(train[features],train["signal"])

print("train a Gradientboost classifier")
gb=GradientBoostingClassifier(n_estimators=120, random_state=3,learning_rate=0.26,subsample=0.7,max_features=34)
gb.fit(train[features[0:-1]],train["signal"])



test_probs= 0.5*ub.predict_proba(test[features[0:-1]])[:, 1]+0.5*gb.predict_proba(test[features[0:-1]])[:, 1]
result = pd.DataFrame({"id": test["id"], "prediction": test_probs})

result.to_csv('UBoost_classifier.csv', index=False, sep=',')