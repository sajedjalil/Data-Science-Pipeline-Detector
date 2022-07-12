
import numpy as np 
import pandas as pd 
test = pd.read_csv('../input/key_2.csv')
train = pd.read_csv('../input/train_2.csv')
test['Page'] = test.Page.apply(lambda a: a[:-11])
train['Visits'] = 0
test = test.merge(train[['Page','Visits']], how='left')
test[['Id','Visits']].to_csv('WTF.csv.gz', index=False, compression='gzip')