import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Loading data
df_test = pd.read_csv('../input/test_users.csv')
id_test = df_test['id']

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx]
    cts += ['other']

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('subAll.csv',index=False)