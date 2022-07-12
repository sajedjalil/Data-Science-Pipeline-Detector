# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np

from sklearn import linear_model
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
df_train =pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

usecols = []
for c in df_train.columns:
    if 'cont' in c:
        usecols.append(c)

x_train = df_train[usecols]
x_test = df_test[usecols]
y_train = df_train['loss']

id_test = df_test['id']


for c in df_train.columns:
    if 'cat' in c:
        df_train[c] = df_train[c].astype('category')
        df_test[c] = df_test[c].astype('category')
        x_train[c + '_numeric'] = df_train[c].cat.codes
        x_test[c + '_numeric'] =  df_test[c].cat.codes
        
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)


# Predict using the trained model 
y_pred = regr.predict(x_test)


sub = pd.DataFrame()
sub['id'] = id_test
sub['loss'] = y_pred
sub.to_csv('lin_regression.csv', index=False)


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.