# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_data_set = pd.read_csv("../input/train.csv")
print(train_data_set.head())
test_df = pd.read_csv("../input/test.csv")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print (train_data_set.shape)
print (test_df.shape)
print (test_df.columns)




cat_data_train = train_data_set.iloc[:,1:117]
cat_data_test = test_df.iloc[:,1:117]
print(cat_data_train.columns)
print(cat_data_test.columns)
cols = cat_data_train.columns
labels = []
print(cols)
for i in range(0,116):
    train = cat_data_train[cols[i]].unique()
    test = cat_data_test[cols[i]].unique()
    print(train)
    labels.append( list(set(train)|set(test)))
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

cats = []
for i in range(0,116):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(train_data_set.iloc[:,i+1])
    feature = feature.reshape(train_data_set.shape[0],1)

    
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)
encoded_cats = np.column_stack(cats)
print(encoded_cats.shape)
encoded_dataset = np.concatenate((encoded_cats,train_data_set.iloc[:,117:].values),axis = 1)
print(encoded_dataset.shape)
from sklearn.linear_model import SGDRegressor
y = encoded_dataset[:,-1]
X = encoded_dataset[:,:-1]


X_train = X[:90000,:]
y_train = y[:90000]
X_cv = X[-20000:,:]
y_cv = y[-20000:]
lr = SGDRegressor(n_iter= 100)

lr.fit(X_train,y_train)
y_hat = lr.predict(X_cv)
loss = abs(y_cv-y_hat).mean()
print("loss: %.f2" % loss)

