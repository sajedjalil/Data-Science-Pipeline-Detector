#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:11:21 2020

@author: stacey
"""

import numpy as np  
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
#from sklearn.linear_model import LogisticRegression
#from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import  metrics
from sklearn.metrics import classification_report,confusion_matrix



data = pd.read_csv('../input/bioresponse/train.csv')

# data.describe()
# data.isnull().sum().sort_values(axes=0, asceding=False)

y = data['Activity']
data.drop(['Activity'], axis=1, inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(data, y, train_size=0.8, test_size=0.2, random_state=1)

pca = PCA(n_components=100)
rf_model = RandomForestClassifier(n_estimators=100, random_state=1)	
#logmodel = LogisticRegression()
#gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)

pipe = Pipeline(steps=[('pca', pca),
                       ('model', rf_model)])

param_grid = {
    'pca__n_components':  [100],
    'model__n_estimators': [400], #[100, 200, 400],
    'model__max_depth': [20] #[10,20,50]
}

search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(X_train, y_train)
print('Best parameters (best score={:.3f}):'.format(search.best_score_))
print(search.best_params_)

#pca.fit(X_train)
#print('Explained variance: ', pca.explained_variance_ratio_)
train_predictions = search.predict(X_train)
valid_predictions = search.predict(X_valid)

log_loss_1 = metrics.log_loss(y_train, train_predictions)
log_loss_2 = metrics.log_loss(y_valid, valid_predictions)
print('Cross-entropy for train data: {:.2f} \n\
Cross-entropy for validation data: {:.2f}'.format(log_loss_1, log_loss_2))


err_train = np.mean(y_train != train_predictions)
err_test  = np.mean(y_valid != valid_predictions)
print(err_train, err_test)

print('Confusion matrix for training set:')
print(confusion_matrix(y_train, train_predictions))
print('\n')
print(classification_report(y_train, train_predictions))


print('Confusion matrix for validation set:')
print(confusion_matrix(y_valid, valid_predictions))
print('\n')
print(classification_report(y_valid, valid_predictions))

X_test = pd.read_csv('../input/bioresponse/test.csv')
test_predictions = search.predict_proba(X_test)
test_df = pd.DataFrame({'MoleculeId': np.arange(1, len(test_predictions)+1),
                        'PredictedProbability': test_predictions[:, 0]
                        })
test_df.to_csv('test_predictions', index=False)
