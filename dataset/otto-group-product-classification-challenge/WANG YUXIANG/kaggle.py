from __future__ import division
import pandas as pd
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

os.system("ls ../input")

data = pd.read_csv("../input/train.csv")
data.pop('id');
data_train, data_test = train_test_split(data, test_size = 0.1)

la = data_train.groupby('feat_83')
for key, item in la:
    tmp = la.get_group(key)

    print (key)
    
    print (tmp[tmp.columns[93]].values)
def getdata(data):
    X = []
    Y = []
    grouped = data.groupby('target')
    for key, item in grouped:
        tmp = grouped.get_group(key)
        X.append(tmp[tmp.columns[0:93]].values)    
        Y.append(tmp[tmp.columns[93]].values)
    return X,Y

