# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nolearn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
## sharing my local cross validation method, 3 folds validation will give about 2 std above lb

def scoreValid(dtrain, dtest):
    #dtrain, dtest = train_test_split(dat_train, test_size=0.3)	
    dtrue = dtest
    dtest = dtest.drop("place_id", 1, )	
    train_rowIDs = np.arange(dtrain.shape[0])
    test_rowIDs = np.arange(dtest.shape[0])	
    dtrain['row_id'] = train_rowIDs
    dtest['row_id'] = test_rowIDs	

    dtrain = dtrain.set_index('row_id')
    dtest = dtest.set_index('row_id')	

    dtrue['row_id'] = test_rowIDs	
    df_outputs = process_grid_cv(dtrain, dtest)
    df_outputs['row_id'] = df_outputs.index
    df_outputs['place_id'] = dtrue['place_id'].tolist()
    def score_cal (true, l1, l2, l3):
        score = 1.0*float(int(true) == int(l1))\
                          + 0.5 * float(int(true) == int(l2)) \
                          + 1/3.0* float(int(true) == int(l3))
        return score
    
    df_outputs['score'] = df_outputs.apply(lambda x : score_cal(x['place_id'], x['l1'], x['l2'], x['l3']), axis=1)
    return df_outputs['score'].mean(), df_outputs