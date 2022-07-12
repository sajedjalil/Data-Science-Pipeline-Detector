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
import matplotlib.pyplot as plt
import random
rnd=57
maxCategories=20

train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
random.seed(rnd)
train.index=train.ID
test.index=test.ID
del train['ID'], test['ID']
target=train.target
del train['target']


#prepare data
traindummies=pd.DataFrame()
testdummies=pd.DataFrame()

for elt in train.columns:
    vector=pd.concat([train[elt],test[elt]], axis=0)

    #count as categorial if number of unique values is less than maxCategories
    if len(vector.unique())<maxCategories:
        traindummies=pd.concat([traindummies, pd.get_dummies(train[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
        testdummies=pd.concat([testdummies, pd.get_dummies(test[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
        del train[elt], test[elt]
    else:
        typ=str(train[elt].dtype)[:3]
        if (typ=='flo') or (typ=='int'):
            minimum=vector.min()
            maximum=vector.max()
            train[elt]=train[elt].fillna(int(minimum)-2)
            test[elt]=test[elt].fillna(int(minimum)-2)
            minimum=int(minimum)-2
            traindummies[elt+'_na']=train[elt].apply(lambda x: 1 if x==minimum else 0)
            testdummies[elt+'_na']=test[elt].apply(lambda x: 1 if x==minimum else 0)
            

            #resize between 0 and 1 linearly ax+b
            a=1/(maximum-minimum)
            b=-a*minimum
            train[elt]=a*train[elt]+b
            test[elt]=a*test[elt]+b
        else:
            if (typ=='obj'):
                list2keep=vector.value_counts()[:maxCategories].index
                train[elt]=train[elt].apply(lambda x: x if x in list2keep else np.nan)
                test[elt]=test[elt].apply(lambda x: x if x in list2keep else np.nan)                
                traindummies=pd.concat([traindummies, pd.get_dummies(train[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
                testdummies=pd.concat([testdummies, pd.get_dummies(test[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
                
                #Replace categories by their weights
                tempTable=pd.concat([train[elt], target], axis=1)
                tempTable=tempTable.groupby(by=elt, axis=0).agg(['sum','count']).target
                tempTable['weight']=tempTable.apply(lambda x: .5+.5*x['sum']/x['count'] if (x['sum']>x['count']-x['sum']) else .5+.5*(x['sum']-x['count'])/x['count'], axis=1)
                tempTable.reset_index(inplace=True)
                train[elt+'weight']=pd.merge(train, tempTable, how='left', on=elt)['weight']
                test[elt+'weight']=pd.merge(test, tempTable, how='left', on=elt)['weight']
                train[elt+'weight']=train[elt+'weight'].fillna(.5)
                test[elt+'weight']=test[elt+'weight'].fillna(.5)
                del train[elt], test[elt]
            else:
                print('error', typ)

#remove na values too similar to v2_na
from sklearn import metrics
for elt in train.columns:
    if (elt[-2:]=='na') & (elt!='v2_na'):
        dist=metrics.pairwise_distances(train.v2_na.reshape(1, -1),train[elt].reshape(1, -1))
        if dist<8:
            del train[elt],test[elt]
        else:
            print(elt, dist)
            
            
train=pd.concat([train,traindummies, target], axis=1)
test=pd.concat([test,testdummies], axis=1)
del traindummies,testdummies


#remove features only present in train or test
for elt in list(set(train.columns)-set(test.columns)):
    del train[elt]
for elt in list(set(test.columns)-set(train.columns)):
    del test[elt]
    

train.to_csv('train.csv', index=False)
