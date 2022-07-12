# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

structures = pd.read_csv('../input/structures.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

def add_structure(df):
    df['type_0'] = df['type'].apply(lambda x: x[0])
    df['type_1'] = df['type'].apply(lambda x: x[1:])    

    df = pd.merge(df, structures, left_on  = ['molecule_name', 'atom_index_0'],
                  right_on = ['molecule_name',  'atom_index'], how='left')
    df = pd.merge(df, structures, left_on  = ['molecule_name', 'atom_index_1'],
                  right_on = ['molecule_name',  'atom_index'], how='left')
    
    df.drop(['id', 'molecule_name', 'atom_index_0','atom_index_1', 'atom_index_x', 'atom_index_y', 'type'], axis=1, inplace=True)

    return df

def decode_labels (train,test):
    for f in ['atom_x', 'atom_y', 'type_1']:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
    return train,test

train=add_structure(train)
test=add_structure(test)
train,test=decode_labels (train,test)

model = LinearRegression()
X_train=train
y_train=X_train['scalar_coupling_constant']
X_train.drop(['scalar_coupling_constant'], axis=1, inplace=True)
model.fit(X_train, y_train)

sample_submission['scalar_coupling_constant'] = model.predict(test)
sample_submission.to_csv('submission002.csv', index = False)