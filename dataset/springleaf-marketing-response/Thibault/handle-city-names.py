import pandas as pd
import numpy as np

train = pd.read_csv("../input/train.csv", nrows=20000)
train = train[['VAR_0237', 'VAR_0274', 'VAR_0200', 'VAR_0241']]

valid_states = set([
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 
    'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 
    'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
])

invalid_states = set(train.VAR_0237.dropna().unique()).difference(valid_states)
print('# Invalid states in VAR_0237: ', len(invalid_states))
invalid_states = set(train.VAR_0274.dropna().unique()).difference(valid_states)
print('# Invalid states in VAR_0274: ', len(invalid_states))
print(invalid_states)
print(train[train.VAR_0274.isin(invalid_states)].groupby('VAR_0274').size())

# Replace with NaN
train.loc[train.VAR_0274.isin(invalid_states), 'VAR_0274'] = np.nan
invalid_states = set(train.VAR_0274.dropna().unique()).difference(valid_states)
print('# Invalid states in VAR_0274: ', len(invalid_states))
