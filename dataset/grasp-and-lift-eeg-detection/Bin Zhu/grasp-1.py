import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from glob import glob
from sklearn.grid_search import GridSearchCV
from sklearn import svm
import os

from sklearn.preprocessing import StandardScaler
 
 
#############function to read data###########

def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#remove id
    labels=labels.drop(['id' ], axis=1)#remove id
    return  clean,labels





################ READ DATA ################################################


X,y=prepare_data_train('../input/train/subj3_series3_data.csv')
X =np.asarray(X.astype(float))
y = np.asarray(y.astype(float))


C_range = 10.0 ** np.arange(-4, 4)
gamma_range = 10.0 ** np.arange(-4, 4)
param_grid = {"gamma": gamma_range.tolist(), "C": C_range.tolist()}
svr = svm.SVC()
grid = GridSearchCV(svr, param_grid)
grid.fit(X[::50,:],y[::50,3])


grid.best_estimator_