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

import xgboost
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, hstack
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import StratifiedKFold, train_test_split

# Load data
print('Loading data...')
train_numeric = pd.read_csv('../input/train_numeric.csv', nrows= 5, header=0)

# 1183747x969   
print('\nTrain numeric columns:')
colnames = train_numeric.columns.values
print('%d columns' % len(colnames)) # 970, 1 id, 1 response, 968 feat
#print('Column names: ' + str(colnames))

response = pd.read_csv('../input/train_numeric.csv', usecols= [969], header=0)
response = np.ravel(response.values)
print('Response ' + str(np.unique(response, return_counts=True))) 
# 1183747x1,  0: 1176868, 1: 6879 (0.0058)
#print(response.head())
print('Train numeric row count: %d' % response.shape[0])
ids = pd.read_csv('../input/train_numeric.csv', usecols= [0], header=0).values
ids = 1.* ids/ids.size

print('\nLoad train numeric to sparse CSC matrix...')
# Number of samples to be used
nrows= 10000 # 1183746 # 700000
print('%d rows used to fit' % nrows)
target = response[:nrows] 
train_num_csc = csc_matrix((nrows, 1))
train_num_csc[:, 0] = ids[:nrows]
# Batch size
batch = 44
for i in range(22):
    print(i)
    bcols = range(i*batch + 1, (i+1)*batch + 1)
    pdb = pd.read_csv('../input/train_numeric.csv', usecols = bcols, nrows=nrows)
    pdb.fillna(0, inplace=True)
    batchcol = csc_matrix(pdb)
    pdb=[0]
    train_num_csc = hstack((train_num_csc, batchcol), format='csc')
    print('Shape traincsc {} batchcol {}'.format(train_num_csc.shape, batchcol.shape))
    batchcol=[0]

print('CSC train num shape: ' + str(train_num_csc.shape))

print('\nSplitting...')
Xtrain, Xval, ytrain, yval = train_test_split(train_num_csc, target,
                                            stratify=target, test_size=0.2)
                                            
print('\nTraining XGBoost (num)...')
clf = xgboost.XGBClassifier(max_depth=5, learning_rate=0.1, max_delta_step=3,
                    scale_pos_weight=170, subsample=0.7, gamma=1, nthread=2)
clf = xgboost.XGBClassifier(max_depth=5, base_score=0.005)
clf.fit(Xtrain, ytrain) #, eval_metric='auc')
print('Train Accuracy: %0.2f' % (100*clf.score(Xtrain, ytrain)))
ypredt = clf.predict(Xtrain)
print('Train Mathews correl: %0.5f' % matthews_corrcoef(ytrain, ypredt))
print('Cross valid Accuracy: %0.2f' % (100*clf.score(Xval, yval)))
ypredv = clf.predict(Xval)
print('Cross valid Mathews correl: %0.5f' % matthews_corrcoef(yval, ypredv))
imp_sort_ids = np.argsort(clf.feature_importances_)[::-1]
sorted_imps = clf.feature_importances_[imp_sort_ids]
ni = 400 # less than 400 feat have import +0
print('%d most important features...'% ni) 
print(sorted_imps[:ni])
print('Total importance %0.5f ' % np.sum(sorted_imps[:ni]))
print(imp_sort_ids[:ni])
print(colnames[imp_sort_ids[:ni]])

numcols = np.where(sorted_imps>0)[0]
print('%d important num cols' % numcols.size)
print(numcols)

train_num_csc = [0]
Xtrain = [0]
Xval = [0]
ytrain = [0]
yval = [0]

print('\nLoad train date to sparse CSC matrix...')
# Number of samples to be used
dnrows= 500000 # 1183746
print('%d rows used to fit' % dnrows)
target = response[:dnrows] 
train_date_csc = csc_matrix((dnrows, 1))
# Batch size
batch = 68
for i in range(17):
    print(i)
    bcols = range(i*batch + 1, (i+1)*batch + 1)
    pdb = pd.read_csv('../input/train_date.csv', usecols = bcols, nrows=dnrows)
    pdb.fillna(0, inplace=True)
    batchcol = csc_matrix(pdb)
    pdb=[0]
    train_date_csc = hstack((train_date_csc, batchcol), format='csc')
    print('Shape traincsc {} batchcol {}'.format(train_date_csc.shape, batchcol.shape))
    batchcol=[0]

print('CSC train date shape: ' + str(train_date_csc.shape))

print('\nSplitting...')
Xtrain, Xval, ytrain, yval = train_test_split(train_date_csc, target,
                                            stratify=target, test_size=0.2)
                                            
print('\nTraining XGBoost (date)...')
clf = xgboost.XGBClassifier(max_depth=5, learning_rate=0.1, max_delta_step=3,
                    scale_pos_weight=170, subsample=0.7, gamma=1, nthread=2)
clf = xgboost.XGBClassifier(max_depth=5, base_score=0.005)
clf.fit(Xtrain, ytrain) #, eval_metric='auc')
print('Train Accuracy: %0.2f' % (100*clf.score(Xtrain, ytrain)))
ypredt = clf.predict(Xtrain)
print('Train Mathews correl: %0.5f' % matthews_corrcoef(ytrain, ypredt))
print('Cross valid Accuracy: %0.2f' % (100*clf.score(Xval, yval)))
ypredv = clf.predict(Xval)
print('Cross valid Mathews correl: %0.5f' % matthews_corrcoef(yval, ypredv))
imp_sort_ids = np.argsort(clf.feature_importances_)[::-1]
sorted_imps = clf.feature_importances_[imp_sort_ids]
ni = 700
print('%d most important features...'% ni) 
print(sorted_imps[:ni])
print('Total importance %0.5f ' % np.sum(sorted_imps[:ni]))
print(imp_sort_ids[:ni])

datecols = np.where(sorted_imps>0)[0]
print('%d important date cols' % datecols.size)
print(datecols)

# pick the best threshold out-of-fold
print('\nPick best threshold...')
predprobs = clf.predict_proba(Xval)[:,1]
thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(yval, predprobs>thr) for thr in thresholds])
plt.plot(thresholds, mcc)
plt.show()
best_threshold = thresholds[np.argmax(mcc)]
auc = roc_auc_score(yval, predprobs)
print('Max MCC: %0.5f, AUC: %0.5f (threshold %0.2f)' % 
                                    (np.max(mcc), auc, best_threshold))

