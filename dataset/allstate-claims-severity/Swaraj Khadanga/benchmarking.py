# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv(r'../input/train.csv')
test = pd.read_csv(r'../input/test.csv')
ids = test.id.values

train['70A'] = 0
train['70B'] = 0

test['70A'] = 0
test['70B'] = 0

for col in list(train.columns.values):
    unique, counts = np.unique(train[col], return_counts=True)
    if len(unique)>2: continue
    train[col].replace('A',1,True)
    train[col].replace('B',0,True)
    test[col].replace('A',1,True)
    test[col].replace('B',0,True)
    #unique, counts = np.unique(train[col], return_counts=True)
    #print(unique)
    train['70A'] += train[col]
    train['70B'] += 1 - train[col]
    test['70A'] += test[col]
    test['70B'] += 1 - test[col]

trainX = train[['70A','70B', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 
'cont6', 'cont7' ,'cont8',
 'cont9', 'cont10' ,'cont11', 'cont12', 'cont13', 'cont14']]
 
trainY = train['loss']

testX = test[['70A','70B', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 
'cont6', 'cont7' ,'cont8',
 'cont9', 'cont10' ,'cont11', 'cont12', 'cont13', 'cont14']]

poly = PolynomialFeatures(degree=2)
trainX = poly.fit_transform(trainX)
testX = poly.fit_transform(testX)

print(trainY)

lm=LinearRegression()
lm.fit(trainX,trainY)

res = lm.predict(testX)

df = pd.DataFrame()
df['id']=ids
df['loss']=res
df.to_csv('search_result_output.csv',index=False)

# Any results you write to the current directory are saved as output.