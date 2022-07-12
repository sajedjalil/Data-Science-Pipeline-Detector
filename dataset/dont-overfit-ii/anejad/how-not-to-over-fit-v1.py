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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)
train.head()
fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3,figsize=(12, 5) );
train[str(0)].hist(ax=ax1);
train[str(1)].hist(ax=ax2);
train[str(2)].hist(ax=ax3);

print(train.isnull().any().any())
#**Check Correlations**
corr = train[train.columns[1:]].corr(method ='spearman')
corr[corr==1]=np.nan
corr[corr==-1]=np.nan
np.min(np.min(corr))
np.max(np.max(corr))
corr['target'].idxmax()
sns.regplot(x='target', y=corr['target'].idxmax(), data=train,ci=None);
corr['target'].idxmin()
sns.regplot(x='target', y=corr['target'].idxmin(), data=train,ci=None);

#Check Descriptive Statistics
descriptive_stats=pd.DataFrame()
descriptive_stats['Mean']=train.mean()
descriptive_stats['Std']=train.std()
descriptive_stats['Skew']=train.skew()
descriptive_stats['Kurtosis']=train.kurtosis()
descriptive_stats=descriptive_stats.drop(['id','target'])
descriptive_stats=descriptive_stats.reset_index();
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(12, 5) );
sns.regplot(x='Mean', y='Std', data=descriptive_stats,ci=None,ax=ax1);
sns.regplot(x='Skew', y='Kurtosis', data=descriptive_stats,ci=None,ax=ax2);

# Modeling
train['target'].value_counts()
n_fold = 20
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
X = train.drop(['id', 'target'], axis=1)
y = train['target']
X_test = test.drop(['id'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
nfolds=20

Cs=np.power(10.0, np.arange(-10, 10))
lrcv = linear_model.LogisticRegressionCV(
    Cs=Cs, penalty='l1', tol=1e-10,class_weight='balanced',cv=nfolds,
    solver='liblinear', n_jobs=4, verbose=1, refit=True,
    max_iter=100)                 
lrcv.fit(X, y) 
print('best score:  '+str(lrcv.scores_[1].mean(axis=0).max()))
plt.plot(lrcv.scores_[1].mean(axis=0))

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
params = {'n_estimators': [200,400,600,1000],
        'learning_rate': [0.05,0.01,0.02,0.05],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
        
xgb = XGBClassifier( objective='binary:logistic', silent=True, nthread=-1)
                    
folds = 5
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)

random_search = RandomizedSearchCV(xgb, param_distributions=params,n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=42)
random_search.fit(X, y)
print(random_search.best_score_)
XGB_Preds=random_search.best_estimator_.predict_proba(X_test)
XGB_Preds=XGB_Preds[:,1]
 
Logistic_Preds=lrcv.predict_proba(X_test)
Logistic_Preds=Logistic_Preds[:,1]


Final_Preds=(0.3*Logistic_Preds+0.7*XGB_Preds)
 
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] =Final_Preds
submission.to_csv('submission_logistic_Regression.csv', index=False)
print('End of Kernel')