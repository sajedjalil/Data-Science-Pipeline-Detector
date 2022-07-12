'''
Feature analysis shows features are less correlated. Therefore Naive Bayes could work well. 
Gaussian Naive Bayes has "no hyperparameter" and requres "no data preprocessing". It's simple
and very fast and we get auc ~ 0.88-0.89 which outperformances many complicated models.
Is also suggests the "Naive" assumption works well in this dataset. 
'''

import pandas 
import numpy as np; np.random.seed(5)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# load data
train_data = pandas.read_csv('../input/train.csv', nrows = None, index_col= 0)
train_data_x = train_data.iloc[:,1:].values
train_data_y = train_data.iloc[:,0].values

# split dataset
x_train, x_test, y_train, y_test = train_test_split(train_data_x, train_data_y, test_size=0.20)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20)
print('train set', x_train.shape, y_train.shape)
print('validation set', x_val.shape, y_val.shape)
print('test set', x_test.shape, y_test.shape)

# fit model
model = GaussianNB()
model.fit(x_train, y_train)

# predict on different sets
y_train_pred = model.predict_proba(x_train)
y_val_pred = model.predict_proba(x_val)
y_test_pred = model.predict_proba(x_test)

# metrics on different sets            
print('auc on train', roc_auc_score(y_train, y_train_pred[:,1]))     
print('auc on val', roc_auc_score(y_val, y_val_pred[:,1]))
print('auc on test', roc_auc_score(y_test, y_test_pred[:,1]))
