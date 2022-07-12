# Forked from Oleg Panichev, to run on v2 training data

# Result should be identical, but I thought there should be an updated script.

# Among other things, this helps to document why we use .97*pred + .03*.11
# or similar numbers that are less magic than they may seem.

import numpy as np
import pandas as pd

data_path = '../input/'
train = pd.read_csv(data_path + 'train_2016_v2.csv')
ss = pd.read_csv(data_path + 'sample_submission.csv')

subm = ss.copy()
mu = train.logerror.mean()

subm['201610'] = mu
subm['201611'] = mu
subm['201612'] = mu

subm['201710'] = mu
subm['201711'] = mu
subm['201712'] = mu

subm.to_csv('submission.csv', index=False, float_format='%.4f')