"""
this kernel is based on:
https://www.kaggle.com/dfrumkin/a-simple-way-to-use-giba-s-features/notebook
https://www.kaggle.com/johnfarrell/giba-s-property-extended-result
https://www.kaggle.com/titericz/the-property-by-giba
https://www.kaggle.com/wentixiaogege/santander-46-features-add-andrew-s-feature-b337d2

and it is just an example of how to use these feautures to replace your submissions,
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm

def get_log_pred(data):
    # Need more features!!! Note that if we use
    features = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', 
                '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 
                'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', 
                '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 
                'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', 
                '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
                '6619d81fc', '1db387535']
    d1 = data[features[:-2]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = data[features[2:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = data[features[0]]
    d2 = d2[d2['pred'] != 0] # Keep?
    d3 = d2[~d2.duplicated(['key'], keep='first')] # Need more features!
    d = d1.merge(d3, how='left', on='key')
    return np.log1p(d.pred).fillna(0)

train = pd.read_csv('../input/santander-value-prediction-challenge/train.csv')

log_pred = get_log_pred(train)
pred_train = np.expm1(log_pred)
have_data = log_pred != 0
print(f'Score = {sqrt(mean_squared_error(np.log1p(train.target[have_data]), log_pred[have_data]))} on {have_data.sum()} out of {train.shape[0]} training samples')


test = pd.read_csv('../input/santander-value-prediction-challenge/test.csv')
log_pred = get_log_pred(test)
pred_test = np.expm1(log_pred)
have_data = log_pred != 0
print(f'Have predictions for {have_data.sum()} out of {test.shape[0]} test samples')


sub = pd.read_csv('../input/santander-46-features-add-andrew-s-feature-b337d2/reduced_set_submission.csv')
for i_iter in tqdm(range(0, len(sub))):
      if pred_test[i_iter]==0:
            pred_test[i_iter] = sub.iloc[i_iter].target
      else:
            pass
del sub['target']
sub['target']=pred_test
print(sub.head())
sub.to_csv('sample_submission.csv', index=False)