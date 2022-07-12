#---------------------------------------------------------------------
#Import the packages
#---------------------------------------------------------------------
import os
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame
import pandas as pd
import xgboost as xgb

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

bst = xgb.train(params, dtrain, 200)
pred = bst.predict(dtest)

df = pd.DataFrame(pred)

#---------------------------------------------------------------------
#Extract features
#---------------------------------------------------------------------

clfs = [
    MultinomialNB(), 
    RandomForestClassifier()]

for clf in clfs:
    clf.fit(x,y)
    df.add(DataFrame(clf.predict_proba(test)))
    
df = df.div(3)
    
l = ['Class_' + str(n) for n in range(1, 10)]
df.columns = l
df.index = range(1, len(df)+1)
df.index.name = 'id'
df.to_csv('out.csv', float_format='%.8f')

    
    
    

    




