# LB = 0.28951

import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/sample_submission_zero.csv')
mem = pd.read_csv('../input/members.csv', usecols=['msno', 'registered_via'])

test = test[['msno']].merge(mem[['msno', 'registered_via']], on='msno', how='left')
train = train[['msno', 'is_churn']].merge(mem[['msno', 'registered_via']], on='msno', how='left')

train.registered_via.fillna(-1, inplace=True)
test.registered_via.fillna(-1, inplace=True)

viamean = train.groupby('registered_via').is_churn.mean().reset_index()
subm = test.merge(viamean, how='left')[['msno','is_churn']]

subm.to_csv('subm_via_mean.csv', index=False)