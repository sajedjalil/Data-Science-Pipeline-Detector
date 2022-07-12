import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import GridSearchCV , cross_val_score
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder , MinMaxScaler ,OneHotEncoder , StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
SDscaler = StandardScaler()

# load data 
data_path = '../input/'
train_X = pd.read_csv(data_path + 'train.csv')
train_Y = pd.read_csv(data_path + 'trainLabels.csv')

# build simple model
rfc = RandomForestClassifier()

rfc.fit(train_X , train_Y)

score = cross_val_score(rfc , train_X , train_Y , n_jobs = -1 , cv = 10).mean()
print("Before tunning AUC score = " ,score)
print("----------------------------------------")

#  GridSearchCV 
from sklearn.model_selection import GridSearchCV


rf = RandomForestClassifier()
param_grid = dict()

grid_search_rf = GridSearchCV(rf, param_grid=dict( ),scoring='accuracy',cv=10).fit(train_X , train_Y)

print('best estimator RandomForest:',grid_search_rf.best_estimator_ , 'Best Score' , grid_search_rf.best_estimator_.score(train_X , train_Y))

rf_best = grid_search_rf.best_estimator_

# build model
rf_best.fit(train_X , train_Y)

print("----------------------------------------")







