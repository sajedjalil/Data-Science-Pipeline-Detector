import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/Train/train.csv')
submission = pd.read_csv('../input/sample_submission.csv')

mean_std = 0.93*train.mean(axis=0) - 0.12*train.std(axis=0)
print(mean_std)

mean_std['adult_males'] = mean_std['adult_males'].mean()+1
mean_std['subadult_males'] = mean_std['subadult_males'].mean()+1
mean_std['adult_females'] = mean_std['adult_females'].mean()-9
mean_std['juveniles'] = mean_std['juveniles'].mean()+3
mean_std['pups'] = mean_std['pups'].mean()

for c in submission.columns:
    if c != 'test_id':
        submission[c] = int(mean_std[c])
submission.to_csv('submission.csv', index=False)
print( submission.head(10) )