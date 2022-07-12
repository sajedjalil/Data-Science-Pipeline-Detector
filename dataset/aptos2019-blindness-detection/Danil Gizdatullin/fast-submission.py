# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# df_sub = pd.read_csv('../input/submission-e-net-best-weight/submission_E_Net_best_weight.csv')
# df_sub = pd.read_csv('../input/sample-submission-0796/submission_0_796.csv')
df_sub = pd.read_csv('../input/submission-best-e-net-kappa/submission_best_kappa.csv')
sample_submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

df_merged = pd.merge(sample_submission, df_sub, on='id_code', how='left')
df_merged.fillna(0, inplace=True)
df_merged['diagnosis'] = df_merged['diagnosis_y']
df_merged = df_merged[['id_code', 'diagnosis']]
df_merged['diagnosis'] = df_merged['diagnosis'].astype('int64')

df_merged.to_csv('submission.csv', index=False)
