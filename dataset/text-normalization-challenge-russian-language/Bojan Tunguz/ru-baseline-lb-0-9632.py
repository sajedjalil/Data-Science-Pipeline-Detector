import pandas as pd
import numpy as np

train = pd.read_csv('../input/ru_train.csv', encoding='utf-8')
train['before'] = train['before'].str.lower()
train['after'] = train['after'].str.lower()
train['after_c'] = train['after'].map(lambda x: len(str(x).split()))
train[~(train['class']=='LETTERS') & (train['after_c']>4)]
train = train.groupby(['before', 'after'], as_index=False)['sentence_id'].count()
train = train.sort_values(['sentence_id','before'], ascending=[False, True])
train = train.drop_duplicates(['before'])
d = {key: value for (key, value) in train[['before', 'after']].values}

test = pd.read_csv('../input/ru_test_2.csv')
test['id'] = test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)
test['before_l'] = test['before'].str.lower()
test['after'] = test['before_l'].map(lambda x: d[x] if x in d else x) #should use same case as original before :)

def fcase(obefore, lbefore, after):
    if lbefore == after:
        return obefore
    else:
        return after
test['after'] = test.apply(lambda r: fcase(r['before'],r['before_l'],r['after']), axis=1)

test[['id','after']].to_csv('submission.csv', index=False)