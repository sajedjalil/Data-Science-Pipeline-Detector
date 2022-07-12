__author__ = 'jonathan'

import os
import numpy as np
import xgboost as xgb
import pandas as pd

os.system("ls ../input")

train = pd.read_csv('../input/train.csv')
train = train.drop('id', axis=1)
y = train['target']
y = y.map(lambda s: s[6:])
y = y.map(lambda s: int(s)-1)
train = train.drop('target', axis=1)
x = train

dtrain = xgb.DMatrix(x.as_matrix(), label=y.tolist())

test = pd.read_csv('../input/test.csv')
test = test.drop('id', axis=1)
dtest = xgb.DMatrix(test.as_matrix())

# print(pd.unique(y.values.ravel()))

params = {'max_depth': 6,
          'objective': 'multi:softprob',
          'eval_metric': 'mlogloss',
          'num_class': 9,
          'nthread': 8}

# cv = xgb.cv(params, dtrain, 50, 3)

# watchlist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(params, dtrain, 50)
# bst.dump_model('dump.raw.txt', 'featmap.txt')

pred = bst.predict(dtest)
# np.savetxt('result.csv', pred, delimiter=',')

df = pd.DataFrame(pred)
l = ['Class_' + str(n) for n in range(1, 10)]
df.columns = l
df.index = range(1, len(df)+1)
df.index.name = 'id'
df.to_csv('out.csv', float_format='%.8f')
