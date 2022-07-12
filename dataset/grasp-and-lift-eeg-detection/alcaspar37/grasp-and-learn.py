# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
from glob import glob
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def prepare_data_train(fname):
    """ read and prepare training data """
    #get data
    data = pd.read_csv(fname)
    #swap to events file
    events_fname = fname.replace('_data','_events')
    #get events results
    labels = pd.read_csv(events_fname)
    clean=data.drop(['id'],axis=1)#remove id
    labels=labels.drop(['id'],axis=1)#remove id
    return clean,labels
    
def prepare_data_test(fname):
    """ read and preprae test data """
    data = pd.read_csv(fname)
    return data
    
subjects = range(1,13)
ids_tot = []
pred_tot = []
    
for subject in subjects:
    y_raw = []
    raw = []
    #read data
    fnames = glob('../input/train/subj%d_series*_data.csv' % (subject))
    for fname in fnames:
        data,labels=prepare_data_train(fname)
        raw.append(data)
        y_raw.append(labels)
    
    X = pd.concat(raw)
    y = pd.concat(y_raw)
    print(type(raw))
    print(type(X))
        
    X_train = np.asarray(X.astype(float))
    y = np.asarray(y.astype(float))
    
    #repeat for train data
    fnames = glob('../input/test/subj%d_series*_data.csv' % (subject))
    test = []
    idx = []
    
    for fname in fnames:
        data=prepare_data_test(fname)
        test.append(data)
        idx.append(np.array(data['id']))
        
    X_test = pd.concat(test)
    ids=np.concatenate(idx)
    print(type(X_test))
    print(type(ids))
        
        
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.