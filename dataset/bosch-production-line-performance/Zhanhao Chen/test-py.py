# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
#from sklearn.metrics import matthews_corrcoef, roc_auc_score
#from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
#import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output

import warnings
warnings.filterwarnings("ignore")

date_chunks = pd.read_csv("../input/train_date.csv", index_col=0, usecols=list(range(5)),chunksize=1000000, dtype=np.float32)
num_chunks = pd.read_csv("../input/train_numeric.csv", index_col=0,
                         usecols=list(range(50)), chunksize=1000000, dtype=np.float32)
#X = pd.concat([pd.concat([dchunk, nchunk], axis=1).sample(frac=0.5)
  #             for dchunk, nchunk in zip(date_chunks, num_chunks)])
for dchunk,nchunk in zip(date_chunks, num_chunks):
    '''print("dchunk")
    print(dchunk)
    print("nchunk")
    print(nchunk)
    print("concat")'''
    X=pd.concat([dchunk, nchunk], axis=1).sample(frac=0.05)
    x=X.values
    print("X")
    y = pd.read_csv("../input/train_numeric.csv", index_col=0, usecols=[0,969], dtype=np.float32).loc[X.index].values.ravel()
    print("y")
    #print(x)
    #print(y)
    clf = XGBClassifier(base_score=0.005)
    clf.fit(X, y)
    print("fit")
    plt.hist(clf.feature_importances_)
    important_indices = np.where(clf.feature_importances_>0)[0]
    print(important_indices)
    break
#print(X.index)
#X = X.values

# Any results you write to the current directory are saved as output.