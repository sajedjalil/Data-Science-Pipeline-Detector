# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn import preprocessing
from subprocess import check_output
from sklearn import linear_model
from sklearn import metrics

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output

#Read the csv train data
data_df = pd.read_csv("../input/train.csv")
m,n = data_df.shape
ftr = data_df.iloc[:,2:n]
test_ids = data_df.iloc[:,0]
y = data_df.pop('species')
y = preprocessing.LabelEncoder().fit(y).transform(y)

#Read test data
data_df1 = pd.read_csv("../input/test.csv")
m1,n1 = data_df1.shape
ftr1 = data_df1.iloc[:,1:n1]



clf = linear_model.LogisticRegression()
clf.fit(ftr,y)
res = clf.predict(ftr1)

submission = pd.DataFrame(res)
submission.to_csv('submission_Ankit1.csv')





