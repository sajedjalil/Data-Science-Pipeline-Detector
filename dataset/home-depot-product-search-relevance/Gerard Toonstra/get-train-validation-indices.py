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
X = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")

''' Returns indices for training set to make a really good set. 
General strategy:
    get last 11.5% of data from train in all cases. This is based on num rows in test that
    do not occur in train. This ensures our training process also doesn't see similar num
    of rows at the end.

    For the remainder, grab one row from train for each product_uid that occurs > 1.
'''
# Group all by product_uid (which disappears), then count aggregated rows in each column
# select only 'id' column, which is a surrogate column name for the rowcount per uid.
trainend = int(len(X)*0.885)
counts = X[:trainend].groupby(['product_uid']).count()[['id']]

# Only care about uid's with counts higher than 1 (do not remove single rows)
counts = counts[counts['id'] > 1]
counts = counts.add_suffix('_Count').reset_index()
valid_product_uids = set(counts['product_uid'].values)

inds = []

allowed_uids = X.loc[X['product_uid'].isin(valid_product_uids)]
# For now, always grab first row of valid product uid.
lastUid = 0

for idx, mrow in allowed_uids.iterrows():
    if lastUid == mrow['product_uid']:
        continue

    lastUid = mrow['product_uid']
    inds.append(idx)

test_inds = inds + list(X[trainend:].index.values)
train_inds = list(X.loc[~X.index.isin(test_inds)].index.values)

print("Train: "+str(len(train_inds))+", test: "+str(len(test_inds)))

