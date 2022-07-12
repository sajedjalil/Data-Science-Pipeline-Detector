# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
from time import time
train.head()

#corr = train.corr()
import matplotlib.pyplot as plt
import seaborn as sns
#fig, ax = plt.subplots(figsize = (18,10))
#sns.heatmap(corr, cmap = sns.diverging_palette(150,275,as_cmap=True), square = True)

#sns.countplot(train['target'])
#sns.kdeplot (kernel density estimation)
#sns.jointplot (scatterplot)
#sns.distplot ( hist)
#sns.boxplot( x= categorical, y = varia)
#sns.violinplot is better than boxplot

#for ind,name in enumerate(train.drop(columns = ['id', 'target']).columns):
#    if ind%10==0:
#        N = []
    
#    N.append(name)
#    if ind%10==9:
#        sns.pairplot(train[N])


sel_train = train.select_dtypes('int').columns.drop('target')[0]
r0 = train[sel_train].min()
r1 = train[sel_train].max()+1
sel_test = test.select_dtypes('int').columns[0]

col_train = [c for c in train.columns if c not in ['id',sel_train, 'target']]
col_test = [c for c in test.columns if c not in ['id',sel_test]]
from sklearn.svm import NuSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
oof = np.zeros(len(train))
pred = np.zeros(len(test))
for i in range(r0,r1):
    print(f'We are at iteration {i}')
    mini_train = train[train[sel_train]==i]
    mini_test = test[test[sel_test]==i]
    idx1 = mini_train.index
    idx2 = mini_test.index
    mini_train.reset_index(drop = True, inplace = True)
    selector = VarianceThreshold(3)
    selector.fit(mini_train[col_train])
    tt = selector.transform(mini_train[col_train])
    tet = selector.transform(mini_test[col_test])
    kf = StratifiedKFold(n_splits = 5, random_state = 2019)
    for train_index, test_index in kf.split(tt, mini_train['target']):
        clf = NuSVC(nu = 0.1, kernel = 'poly', degree = 4, probability = True, gamma = 'auto')
        clf.fit(tt[train_index,:], mini_train.loc[train_index]['target'])
        oof[idx1[test_index]] += clf.predict_proba(tt[test_index,:])[:,1]
        pred[idx2] += clf.predict_proba(tet)[:,1] / kf.n_splits
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(train['target'], oof)
print(f'Cv score: {auc}')

sub= pd.read_csv('../input/sample_submission.csv')
sub['target'] = pred
sub.to_csv('submission.csv',index=False)
#import scipy.stats as stats
#fig, ax = plt.subplots(figsize = (12,8))
#stats.probplot(train.iloc[:,1], plot =plt)
    
#from sklearn.model_selection import train_test_split

#Xy_train, Xy_test = train_test_split(train, test_size = 0.2, random_state = 2019)
#X_train = Xy_train.drop(columns = [sel_train,'target', 'id'])
#y_train = Xy_train['target']

#X_test = Xy_test.drop(columns = [sel_train,'target', 'id'])
#y_test = Xy_test['target']

#from sklearn.linear_model import LogisticRegression


#initial_time = time()
#Model_bench_log = LogisticRegression(random_state = 2019, verbose = 1, n_jobs = -1, solver = 'lbfgs')
#Model_bench_log.fit(X_train,y_train)
#print(f'Fitting done in { time()-initial_time}')
#print(f'The cross-validation score for the logistic model is {Model_bench_log.score(X_test,y_test)}')

#from sklearn.svm import LinearSVC

#initial_time = time()
#clf = LinearSVC(dual = False, verbose = 1)
#clf.fit(X_train, y_train)
#print(f'Fitting executed in {time() - initial_time}')
#print(f'The cross-validation score is {clf.score(X_test, y_test)}')

