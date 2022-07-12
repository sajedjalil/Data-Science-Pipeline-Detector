import pandas as pd
import numpy as np

train = pd.read_csv("../input/train_1.csv").fillna(0)
test = pd.read_csv("../input/key_1.csv")

for col in train.columns[1:]:
    train[col] = pd.to_numeric(train[col],downcast='integer')

test['Page'] = test.Page.apply(lambda a: a[:-11])

# note that this is LEAKY due to not dropping an appropiate amount of events to simulate test
train['Visits'] = train[train.columns[-14:]].median(axis=1, skipna=True)

test = test.merge(train[['Page','Visits']], how='left')
test.loc[test.Visits.isnull(), 'Visits'] = 0

test[['Id','Visits']].to_csv('med.csv', index=False)