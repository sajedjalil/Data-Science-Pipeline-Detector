# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import random
random.seed(2016)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

for c in list(train.columns):
    print('Column '+ str(c))
    print(train[c].describe())
    u = train[c].unique()
    print(len(u))
    print(u)
    print('')
for c in list(test.columns):
    print('Column '+ str(c))
    print(test[c].describe())
    u = test[c].unique()
    print(len(u))
    print(u)
    print('')


