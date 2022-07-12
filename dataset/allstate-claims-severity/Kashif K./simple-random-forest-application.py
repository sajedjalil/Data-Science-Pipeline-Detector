# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
import pylab as pl
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_csv('../input/train.csv')
df1 = pd.read_csv('../input/test.csv')

Xi = df[df.columns[1:132]]


for x in range(1, 117):
    df1['cat_ord%d'%(x)]  = pd.Categorical(df1['cat%d'%(x)]).codes
    
Xo_test = df1[['cat_ord%d'%(x) for x in range(1, 117)]] 
Xo_1 = df1[df1.columns[117:131]]
Xo_0 = df1[df1.columns[0]]
Xo_test = pd.concat([Xo_test, Xo_1], axis=1)

for x in range(1, 117):
    df['cat_ord%d'%(x)]  = pd.Categorical(df['cat%d'%(x)]).codes


Xo = df[['cat_ord%d'%(x) for x in range(1, 117)]]  
y = df[['loss']]

y_mean = y.mean()

X1 = df[df.columns[117:132]]
X = pd.concat([Xo, X1], axis=1)
X['Mean'] = X['loss'].mean()
X['Severe'] = np.where(X['loss'] >= X['Mean'], 1, 0)
X.drop(X.columns[[131]], axis=1, inplace=True)

features = list(X.columns[:130])
features

y = X['loss']
Xo = df[features]
test_id = list(Xo_0)
sdf = Xo_test.to_sparse()

clf = RandomForestRegressor(n_estimators=50)
clf = clf.fit(Xo, y)
#clf.predict(sdf)
#print clf.predict(sdf)

rows = zip(Xo_0, clf.predict(sdf))


with open('submission_rfr3.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'loss'])
    
    for row in rows:
        writer.writerow(row)


# Any results you write to the current directory are saved as output.