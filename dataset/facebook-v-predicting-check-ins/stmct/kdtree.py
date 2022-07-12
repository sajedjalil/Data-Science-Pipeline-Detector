# Read data
import numpy as np
import pandas as pd

types = {'row_id': np.dtype(int),
         'x': np.dtype(float),
         'y' : np.dtype(float),
         'accuracy': np.dtype(int),
         'place_id': np.dtype(int) }

train = pd.read_csv('../input/train.csv', dtype=types, index_col=0)
test = pd.read_csv('../input/test.csv', dtype=types, index_col=0)
sample_submission = pd.read_csv('../input/sample_submission.csv', index_col=0)

# Calc means of place coords
locations = train.groupby('place_id')[['x', 'y']].mean()

# Find 3 neighbors w/ KDTree
from scipy import spatial

tree = spatial.cKDTree(locations)
neighbors = tree.query(test[['x', 'y']], k=3)[1]

# Create submission
preds = []
for neighbor in neighbors:
    preds.append(' '.join(str(s) for s in locations.index[neighbor]))

sample_submission['place_id'] = preds
sample_submission.to_csv('submission.csv')