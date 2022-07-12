# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import SGDClassifier
import time

# Based on http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html
def get_matthews_corrcoef(y_test, predictions):
    return matthews_corrcoef(y_test, predictions)

def progress(cls_name, stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = "%10s classifier : \t" % cls_name
    s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
    # s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
    s += "accuracy: %(accuracy).3f " % stats
    s += "matthews_corrcoef: %(matthews_corrcoef).3f " % stats
    s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
    return s


def oo_learn():
    file_path = '../input/train_numeric.csv'
    X=None
    y=None
    ids=None
    print("Online Learning from '{}'".format(file_path))
    chunks = []
    X_chunks = []
    y_chunks = []
    ids_chunks = []
    c=0
    cls = SGDClassifier(n_jobs=7, penalty='l1', random_state=42, learning_rate ='optimal' )
    cls_stats = {}
    
    cls_name='SGDClassifier'
    stats = {'n_train': 0, 'n_train_pos': 0,
             'accuracy': 0.0, 'matthews_corrcoef': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
             'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
    cls_stats[cls_name] = stats
    chunksize=25000
    all_classes = np.array([0, 1])
    tick = time.time()
    total_vect_time = 0.0
    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype='float32'):
        print("Chunk {}".format(c))
        ids_chunks.append(chunk.ix[:,0])
        y = chunk.ix[:,-1]
        X = chunk.ix[:,1:chunk.shape[1]-1]
        X = X.replace(np.nan,0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        cls.partial_fit(X_train, y_train, classes=all_classes)
        c+=1
        # accumulate test accuracy stats
        cls_stats[cls_name]['total_fit_time'] += time.time() - tick
        cls_stats[cls_name]['n_train'] += X_train.shape[0]
        cls_stats[cls_name]['n_train_pos'] += sum(y_train)
        tick = time.time()
        cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
        predictions = cls.predict(X_test)
        cls_stats[cls_name]['matthews_corrcoef'] = get_matthews_corrcoef(y_test, predictions)
        cls_stats[cls_name]['prediction_time'] = time.time() - tick
        acc_history = (cls_stats[cls_name]['accuracy'], cls_stats[cls_name]['n_train'])
        cls_stats[cls_name]['accuracy_history'].append(acc_history)
        run_history = (cls_stats[cls_name]['accuracy'], total_vect_time + cls_stats[cls_name]['total_fit_time'])
        cls_stats[cls_name]['runtime_history'].append(run_history)

        print(progress(cls_name, cls_stats[cls_name]))

    #print(cls.coef_ )
    
oo_learn()
    