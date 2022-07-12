import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter

df_train = pd.read_csv('../input/gender_age_train.csv')
occurs = Counter(df_train['group'])
sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns:
    if c != 'device_id':
        sub[c] = occurs[c] / len(df_train.index)
sub.to_csv('beat_the_benchmark.csv', index=False)