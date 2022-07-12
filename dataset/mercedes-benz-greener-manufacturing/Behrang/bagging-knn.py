# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
train_np=train.as_matrix()
test_np=test.as_matrix()

x_train=train_np[:,2:]
y_train=train_np[:,1]

x_test=test_np[:,1:]

kn1= neighbors.KNeighborsRegressor(n_neighbors=10,weights='uniform')
kn2= neighbors.KNeighborsRegressor(n_neighbors=10,weights='distance')
bgg= BaggingRegressor(kn1,n_estimators=10,max_samples=0.7,max_features=0.9,verbose=0)#, max_features=0.5

bgg.fit(x_train,y_train)
print(bgg.score(x_train,y_train))
print(r2_score(bgg.predict(x_train), y_train))

y_pred = bgg.predict(x_test)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('submission_baseLine.csv', index=False)