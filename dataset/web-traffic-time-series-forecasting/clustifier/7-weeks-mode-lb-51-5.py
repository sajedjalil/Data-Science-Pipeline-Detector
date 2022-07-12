import pandas as pd
import numpy as np
from collections import Counter

def mode(df):
    c = Counter(df.values)
    return c.most_common()[0][0]

train = pd.read_csv("../input/train_1.csv")
test = pd.read_csv("../input/key_1.csv")

test['Page'] = test.Page.apply(lambda a: a[:-11])

train['Visits'] = train[train.columns[-49:]].apply(mode, axis=1)

test = test.merge(train[['Page','Visits']], how='left')
test.loc[test.Visits.isnull(), 'Visits'] = 0

test[['Id','Visits']].to_csv('mode.csv', index=False)