import pandas as pd
import random
from random import shuffle

random.seed(2)
s = pd.read_csv('../input/sample_submission.csv')
print(s.head(1))
def r(x):
    l = str(x).split(' ')
    shuffle(l)
    return ' '.join(l)
s['ad_id'] = s['ad_id'].apply(r)
print(s.head(1))
s.to_csv('s.csv', index=False)