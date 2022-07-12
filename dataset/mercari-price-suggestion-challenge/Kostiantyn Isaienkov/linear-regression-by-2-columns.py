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

np.random.seed(42)

train = pd.read_csv('../input/train.tsv', sep='\t')


price = train['price']
train = train.drop(['train_id', 'item_description', 'category_name', 'brand_name', 'price', 'name'], axis = 1)

test = pd.read_csv('../input/test.tsv', sep='\t')

id = test['test_id']
test = test.drop(['test_id', 'item_description', 'category_name', 'brand_name', 'name'], axis = 1)

from sklearn.linear_model import LinearRegression


train.fillna(0)
test.fillna(0)


lr = LinearRegression()

lr.fit(train, price)


pred = lr.predict(test)
pred = pd.DataFrame(pred, columns=['price'])

ans = pd.concat([id, pred], axis = 1)

ans.to_csv('sub.csv', index = False)