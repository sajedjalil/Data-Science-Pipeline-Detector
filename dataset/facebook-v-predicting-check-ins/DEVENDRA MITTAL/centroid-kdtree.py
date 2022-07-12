import pandas as pd
import time
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.neighbors import KDTree


# Input data files are available in the "../input/" directory.

train = pd.read_csv("../input/train.csv", header=0)
print('train file read complete')

centroid = train.groupby('place_id').agg({'x': np.median, 'y': np.median})
print('centroid compute complete')

tree = KDTree(centroid[['x', 'y']])
print('tree created')

test = pd.read_csv("../input/test.csv", header=0)
print('test file read complete')

_, points = tree.query(test[['x', 'y']], k=1)
print('query complete')

p1 = [x[0] for x in points]
# print(centroid.iloc[p1])
test['place_id'] = centroid.iloc[p1].index
test = test[['row_id', 'place_id']]
test.to_csv("centroid_KD_Median_submission.csv", index=False)
