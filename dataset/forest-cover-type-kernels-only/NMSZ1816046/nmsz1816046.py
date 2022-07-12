# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
 # linear algebra
 # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
train_df = pd.read_csv('../input/train.csv')
validation_df = pd.read_csv('../input/test.csv',error_bad_lines = False)
n_columns = len(train_df.columns)
feature_column_names = train_df.columns[0:n_columns - 1]
label_column_name = train_df.columns[n_columns - 1]
X_train=train_df[feature_column_names]
Y_train=train_df[label_column_name]
X_dv = X_train.values
Y_dv = Y_train.values
for i in range(3):
    X_dv = np.vstack((X_dv,X_dv))
    Y_dv = np.hstack((Y_dv,Y_dv))
clf = RandomForestClassifier(n_estimators=890,random_state=14,max_features=17)#0.868716 890 14 17
s = clf.fit(X_dv,Y_dv)
y = s.predict(validation_df)
y_df = pd.DataFrame(y,columns=['Cover_Type'])
arr = validation_df['Id']
result = pd.concat([arr,y_df],axis=1, join='inner')
result.to_csv('result.csv',index=False,header=True)