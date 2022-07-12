
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import log

train = pd.read_csv('../input/Train/train.csv')
submission = pd.read_csv('../input/sample_submission.csv')

mean = train.mean(axis=0)

for c in submission.columns:
    if c != 'test_id':
        submission[c] = int(mean[c])

submission.to_csv('submission.csv', index=False)

print(train.describe())
