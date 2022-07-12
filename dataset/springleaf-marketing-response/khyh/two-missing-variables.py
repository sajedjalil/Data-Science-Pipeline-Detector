import pandas as pd
import numpy as np

train = pd.read_csv("../input/train.csv", nrows=100)
train.drop(['ID', 'target'], axis=1, inplace=True)

#print(train.columns)
# = ['VAR_0001', 'VAR_0002', 'VAR_0003', ... 'VAR_1933', 'VAR_1934']

columns = list(train.columns)

for i in range(1, 1935):
    # i ~ 1, 2, ... 1934
    varname = 'VAR_{:04d}'.format(i)
    if not (varname in columns):
        print(varname)

# contrary to my expectation,
# VAR_0218 and VAR_0240 does not exists!!!