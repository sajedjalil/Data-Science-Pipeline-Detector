# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import os
import numpy as np


train = pd.read_csv('../input/train.csv', low_memory=True)
target = 'Demanda_uni_equil'     
columns = ['Cliente_ID', 'Producto_ID']
medians = train.groupby( columns )[target].median()
global_median = np.median(train[target])
del train
test = pd.read_csv('../input/test.csv', low_memory=True)
medians = medians.reset_index()

test_merge = pd.merge( test, medians, on = columns, how = 'left' )
del test
del medians
test_merge[target].fillna(global_median, inplace=True)
sub_file = os.path.join('subm', 'submission.csv')
        
test_merge[[ 'id', target ]].to_csv(sub_file, index = False )
