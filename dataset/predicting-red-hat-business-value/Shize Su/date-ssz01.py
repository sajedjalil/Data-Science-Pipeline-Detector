import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#%matplotlib inline

train = pd.read_csv('../input/act_train.csv', parse_dates=['date'])
test = pd.read_csv('../input/act_test.csv', parse_dates=['date'])
ppl = pd.read_csv('../input/people.csv', parse_dates=['date'])

df_train = pd.merge(train, ppl, on='people_id')
df_test = pd.merge(test, ppl, on='people_id')
del train, test, ppl


for d in ['date_x', 'date_y']:
    print('Start of ' + d + ': ' + str(df_train[d].min().date()))
    print('  End of ' + d + ': ' + str(df_train[d].max().date()))
    print('Range of ' + d + ': ' + str(df_train[d].max() - df_train[d].min()) + '\n')
    
    


tr=df_train[['activity_id','date_x', 'date_y','outcome']]
tr.to_csv('tr_date.csv', index=False)


sub=df_test[['activity_id','date_x', 'date_y']]
sub.to_csv('sub_date.csv', index=False)

