# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
a = pd.read_csv("../input/train.csv")
b = pd.read_csv("../input/test.csv")
c = pd.concat([a,b], ignore_index=True)

catdf = pd.DataFrame()
tmp1 = c.loc[:,c.columns.str.startswith('cat')].apply(pd.factorize)

for index, row in tmp1.iteritems():
    catdf[index] = row[0]

tmp2 = c.drop(c.loc[:,c.columns.str.startswith('cat')],axis=1)
tmp3 = tmp2.join(catdf)

train = tmp3.loc[:188317,]
test = tmp3.loc[188318:,]

X_train = train.drop('loss',1)
Y_train = np.array(train.loss, int)

from xgboost import XGBRegressor
xgb1 = XGBRegressor(max_depth=8, learning_rate=0.07, min_child_weight=1,
                    gamma = 0, scale_pos_weight=1,n_estimators=300,
                                objective="reg:linear", subsample=1,
                                colsample_bytree=1, seed=1234)
xgb1.fit(X_train, Y_train)
loss = xgb1.predict(test.drop('loss',1))
result = pd.DataFrame({'id':np.array(test.id),
                       'loss':loss})
                       
f = open('result.csv', 'w')
f.writelines(result.to_csv(index=False))
f.close
# Any results you write to the current directory are saved as output.