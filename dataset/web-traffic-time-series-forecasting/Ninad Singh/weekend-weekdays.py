import pandas as pd
import numpy as np

train = pd.read_csv("../input/train_1.csv")
train_flattened = pd.melt(train[list(train.columns[-49:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')
train_flattened['weekend'] = ((train_flattened.date.dt.dayofweek) // 5 == 1).astype(float)

test = pd.read_csv("../input/key_1.csv")
test['date'] = test.Page.apply(lambda a: a[-10:])
test['Page'] = test.Page.apply(lambda a: a[:-11])
test['date'] = test['date'].astype('datetime64[ns]')
test['weekend'] = ((test.date.dt.dayofweek) // 5 == 1).astype(float)

train_page_per_dow = train_flattened.groupby(['Page','weekend']).median().reset_index()

test = test.merge(train_page_per_dow, how='left')
test.loc[test.Visits.isnull(), 'Visits'] = 0

test[['Id','Visits']].to_csv('mad.csv', index=False)