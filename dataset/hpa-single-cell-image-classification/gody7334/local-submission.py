# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def isNaN(num):
    return num != num

sample_submission = pd.read_csv('../input/hpa-single-cell-image-classification/sample_submission.csv')
local_df = pd.read_csv('../input/hpascicsubmitfile/local_submission.csv',index_col=0)

for i, row in sample_submission.iterrows():
    if row['ID'] in local_df.index:
        if isNaN(local_df.loc[row['ID']].PredictionString): continue
        sample_submission.PredictionString.loc[i] = local_df.loc[row['ID']].PredictionString

sample_submission.to_csv('submission.csv', index=False)




