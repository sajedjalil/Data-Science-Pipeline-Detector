# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os



train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
test_df = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')

# Check shapes of datasets
print(train_df.shape)
print(test_df.shape)



if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):
    submission = pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv')
    submission['isup_grade'] = np.array([0, 3, 4])
    submission.to_csv('submission.csv',index=False)
else:
    print("Train only. Saving sample submission")
    submission = pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv')
    submission['isup_grade'] = np.array([0, 0, 0])
    submission.to_csv('submission.csv',index=False)
    
    
