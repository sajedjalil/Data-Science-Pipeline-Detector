# coding: utf-8

import pandas as pd

train = pd.read_csv('../input/train.csv', index_col='id')
test = pd.read_csv('../input/test.csv', index_col='test_id')

output = 'naive.csv'


pred = train.is_duplicate.mean()

test['is_duplicate'] = pred * 0.3

test[['is_duplicate']].to_csv(output)

