import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/Train/train.csv')
submission = pd.read_csv('../input/sample_submission.csv')

mean_std = 0.94*train.mean(axis=0) - 0.12*train.std(axis=0)
print(mean_std)

mean_std['adult_males'] = 5
mean_std['subadult_males'] = 4
mean_std['adult_females'] = 26
mean_std['juveniles'] = 15
mean_std['pups'] = 11

for c in submission.columns:
    if c != 'test_id':
        submission[c] = int(mean_std[c])
submission.to_csv('submission.csv', index=False)
print( submission.head(10) )
