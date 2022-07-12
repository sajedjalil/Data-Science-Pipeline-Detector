import pandas as pd
import numpy as np
import gc

dtypes = {'ad_id': np.float32, 'clicked': np.int8}

print('Load train clicks...')
train = pd.read_csv("../input/clicks_train.csv", usecols=['ad_id','clicked'], dtype=dtypes)
print(train.head())
print(train.shape)

print('Get advert likelihood')
ad_likelihood = train.groupby('ad_id').clicked.agg(['count','sum','mean']).reset_index()
print(ad_likelihood.shape)
print(ad_likelihood.head())
M = train.clicked.mean()
print('Mean likelihood: %0.5f' % M)
del train
gc.collect()

print('Append likelihood series')
ad_likelihood['likelihood'] = (ad_likelihood['sum'] + 12*M) / (12 + ad_likelihood['count'])
print(ad_likelihood.shape)
print(ad_likelihood.head())

print('Load test clicks...')
test = pd.read_csv("../input/clicks_test.csv")
print(test.head())
print('Merge with likelihoods and fill NA...')
test = test.merge(ad_likelihood, how='left')
test.likelihood.fillna(M, inplace=True)

print('Sort values...')
test.sort_values(['display_id','likelihood'], inplace=True, ascending=False)
print('Group by display...')
subm = test.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()
print('Submit')
subm.to_csv("subm.csv", index=False)

del test
gc.collect()