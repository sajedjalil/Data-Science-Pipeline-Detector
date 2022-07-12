from time import time
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
#%matplotlib

# Number of cores to use to perform parallel fitting of the forest model
n_jobs = -1 # all the cores
# read in data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# create true labels (y) and feature vector (train)
y = train.Response.values
ids = test.Id.values
train = train.drop(['Response','Id'], axis=1)
test = test.drop(['Id'], axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

# convert string categoricals into numeric
for f in train.columns:
    if train[f].dtype=='object':
        #print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        #lbl.fit(list(train[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
        
# training data shape
train.shape

# normalize
#scaler = preprocessing.StandardScaler().fit(train) # scales, centers, normalizes
#train = scaler.transform(train)
#test = scaler.transform(test)

# classifiers
#forest = RandomForestClassifier(n_estimators=100,
#                              n_jobs=n_jobs,
#                              random_state=0)

#extra = ExtraTreesClassifier(n_estimators=100,
#                              n_jobs=n_jobs,
#                              random_state=0,
#                             class_weight = 'balanced')

param = {'max_depth':10, 'eta':10**-2, 'silent':1, 'min_child_weight':1, 'subsample' : 0.7 ,"early_stopping_rounds":10,
          "objective"   : "reg:linear",'eval_metric': 'rmse','colsample_bytree':0.8}

xg = XGBClassifier(param)

#eclf = VotingClassifier(estimators = [('rf',forest),('xg',xg),('et',extra)],voting='soft')

#eclf.fit(train, y)

xg.fit(train, y)

# do predictions on test set
y_pred_submit = xg.predict(test)
import csv

n_ids=len(ids)

prediction_file = open("prudential_predictions_xgb.csv", 'w')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["Id","Response"])
for i in range(0,n_ids):
    prediction_file_object.writerow([ids[i],y_pred_submit[i]])
    
prediction_file.close()