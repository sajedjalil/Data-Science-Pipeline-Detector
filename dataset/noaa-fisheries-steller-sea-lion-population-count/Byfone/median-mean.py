
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import log

train = pd.read_csv('../input/Train/train.csv')
submission = pd.read_csv('../input/sample_submission.csv')

median = train.median(axis=0)
mean = train.mean(axis=0)

for c in submission.columns:
    if c != 'test_id':
        submission[c] = int( (median[c]-mean[c])/(log(median[c]+1e-9)-log(mean[c]+1e-9)) )
submission.to_csv('submission.csv', index=False)