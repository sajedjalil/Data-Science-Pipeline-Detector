import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
scalar_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv')
structures = pd.read_csv('../input/structures.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

train = train[train.atom_index_0!=0] # Remove rows with atom_index not present in test set

train = pd.merge(train, structures, left_on  = ['molecule_name', 'atom_index_0'],
                  right_on = ['molecule_name',  'atom_index'], how='left')
train = pd.merge(train, structures, left_on  = ['molecule_name', 'atom_index_1'],
                  right_on = ['molecule_name',  'atom_index'], how='left')
test = pd.merge(test, structures, left_on  = ['molecule_name', 'atom_index_0'],
                  right_on = ['molecule_name',  'atom_index'], how='left')
test = pd.merge(test, structures, left_on  = ['molecule_name', 'atom_index_1'],
                  right_on = ['molecule_name',  'atom_index'], how='left')

train = pd.merge(train, scalar_coupling_contributions, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])
test = pd.merge(test, scalar_coupling_contributions, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

lbl = LabelEncoder()
for i in range(4):
    train['type'+str(i)] = lbl.fit_transform(train['type'].map(lambda x: str(x)[i]))
    test['type'+str(i)] = lbl.transform(test['type'].map(lambda x: str(x)[i]))

y = train.scalar_coupling_constant.values
train.drop(['id', 'molecule_name', 'atom_index_0','atom_index_1', 'atom_index_x', 'atom_index_y', 'scalar_coupling_constant', 'type'], axis=1, inplace=True)
test.drop(['id', 'molecule_name', 'atom_index_0','atom_index_1', 'atom_index_x', 'atom_index_y', 'type'], axis=1, inplace=True)

def get_dummies(train, test):
    encoded = pd.get_dummies(pd.concat([train,test], axis=0))
    train_rows = train.shape[0]
    train = encoded.iloc[:train_rows, :]
    test = encoded.iloc[train_rows:, :] 
    return train,test

train, test = get_dummies(train, test)

X = train.loc[:,['x_x', 'y_x', 'z_x', 'x_y', 'y_y', 'z_y', 'type0', 'type1',
       'type2', 'type3', 'atom_x_H', 'atom_y_C', 'atom_y_H', 'atom_y_N']]
X_test = test.loc[:,['x_x', 'y_x', 'z_x', 'x_y', 'y_y', 'z_y', 'type0', 'type1',
       'type2', 'type3', 'atom_x_H', 'atom_y_C', 'atom_y_H', 'atom_y_N']]

dt = KNeighborsRegressor()
test['fc'] = dt.fit(X, train.fc.values).predict(X_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train.fc.values.reshape(-1, 1), y)
sample_submission['scalar_coupling_constant'] = lr.predict(test.fc.values.reshape(-1, 1))
sample_submission.to_csv('bench.csv', index = False)