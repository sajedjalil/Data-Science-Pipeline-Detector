# Read data
import numpy as np
import pandas as pd
from scipy import spatial
from collections import Counter

types = {'row_id': np.dtype(int),
         'x': np.dtype(float),
         'y' : np.dtype(float),
         'accuracy': np.dtype(int),
         'place_id': np.dtype(int) }

train = pd.read_csv('../input/train.csv', dtype=types, index_col=0)
test = pd.read_csv('../input/test.csv', dtype=types, index_col=0)


# use latest 10% data
train = train[train.time >= train.time.quantile(0.9)]

# add columns
def add_cols(df):
    df['hour'] = df.time//60%24
    # df['weekday'] = df.time//60//24%7
add_cols(train)
add_cols(test)

# Find 50 neighbors w/ KDTree
tree = spatial.cKDTree(train[['x', 'y', 'hour']])
neighbors = tree.query(test[['x', 'y', 'hour']], k=50)[1]

# Create submission
train_place_id = train.place_id.values

preds = []
for i in range(len(test)):
    if i % 100000 == 0:
        print(i, end=', ')
    preds.append(' '.join(str(s) for s in list(zip(*Counter(train_place_id[neighbors[i]]).most_common(3)))[0]))

preds = pd.DataFrame({'place_id': preds})
preds.index.name = 'row_id'
preds.to_csv('submission.csv')