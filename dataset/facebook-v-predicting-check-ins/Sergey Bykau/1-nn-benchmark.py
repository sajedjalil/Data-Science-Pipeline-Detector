import pandas as pd
from sklearn.neighbors import KDTree


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

tree = KDTree(train[['x', 'y']])
_, ind = tree.query(test[['x','y']], k=1)
ind1 = [x[0] for x in ind]
test['place_id'] = train.iloc[ind1].place_id.values
test[['row_id', 'place_id']].to_csv('submission.gz', index=False, compression='gzip')
