#Naive submission using 3 nearest neighbours with KD-tree search.
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import spatial

random.seed(2016)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print('Loading data...(Only 10,000 rows)')
data = pd.read_csv('../input/train.csv', nrows=10000)
test = pd.read_csv('../input/test.csv', nrows=10000)
X_test = np.column_stack((test['x'],test['y']))
row_ids = test['row_id']

print('Finding average x and y for each place id...')
df = data.groupby('place_id')['x','y'].mean()

print('Building KD Tree...')
tree = spatial.KDTree(df[['x','y']].as_matrix(), leafsize=20)

print('Finding nearest neighbours...')
def get_3NN(x):
    point = [x['x'],x['y']]
    distance, points = tree.query(point, k=3)
    place_ids = [df.iloc[i].name for i in points]
    return " ".join(map(str, place_ids))
test['place_id'] = test.apply(get_3NN, axis=1)

print('Outputting...')
test.to_csv('submission.csv',columns=['row_id','place_id'], header=True, index=False)
print('Done.')