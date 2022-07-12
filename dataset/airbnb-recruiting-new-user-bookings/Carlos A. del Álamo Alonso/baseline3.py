import numpy as np
import pandas as pd



#Loading data
train = pd.read_csv('../input/train_users.csv')
test = pd.read_csv('../input/test_users.csv')


result = []
for index, row in test.iterrows():
    if isinstance(row['date_first_booking'], float):
        result.append([row['id'], 'US'])
        result.append([row['id'], 'other'])
        result.append([row['id'], 'FR'])
        result.append([row['id'], 'IT'])
        result.append([row['id'], 'GB'])
    else:
        result.append([row['id'], 'NDF'])
        
pd.DataFrame(result).to_csv('sub.csv', index = False, header = ['id', 'country'])