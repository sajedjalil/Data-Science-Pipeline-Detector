'''
This benchmark is a template for a GridSearch of parameters for a RF

Use the data loading of the Devin Xgboost benchmark script
https://www.kaggle.com/devinanzelmo/liberty-mutual-group-property-inspection-prediction/xgboost-benchmark-0-38019

@author juandoso

'''

import pandas as pd
import numpy as np 
from sklearn import preprocessing

#load train and test 
train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
    if type(train[1,i]) is str:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

train = train.astype(float)
test = test.astype(float)

from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
param_grid = {'n_estimators': [50, 100]}
model = GridSearchCV(RandomForestRegressor(), param_grid)
model = model.fit(train,labels)
print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

preds = model.predict(test)


#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('rf_benchmark.csv')