import numpy as np
import pandas as pd

data_path = '../input/'
train = pd.read_csv(data_path + 'train_2016.csv')
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