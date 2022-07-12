# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# loading train data
traindata = pd.read_csv('../input/train.csv')
x_tr = traindata.values[:, 2:]
y_tr = traindata.values[:, 1]
le = LabelEncoder().fit(traindata['species'])
scaler = StandardScaler().fit(x_tr)
x_tr = scaler.transform(x_tr)
#loading test data
testdata = pd.read_csv('../input/test.csv')
x_test = testdata.drop(['id'], axis=1).values
x_test = scaler.transform(x_test)
test_ids = testdata.pop('id')
#Start learning
random_forest = RandomForestClassifier(n_estimators=500)
random_forest.fit(x_tr, y_tr)
#make permission 
y_pred = random_forest.predict_proba(x_test)
#submission
submission = pd.DataFrame(y_pred, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')