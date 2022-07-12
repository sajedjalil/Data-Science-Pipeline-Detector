# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import scipy
from sklearn.linear_model import LogisticRegression
import optuna



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


train=pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')
test=pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')
submission=pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv', index_col='id')
Ytrain=train['target']
train=train[list(test)]
all_data=pd.concat((train, test))
print(train.shape, test.shape, all_data.shape)

encoded=pd.get_dummies(all_data, columns=all_data.columns, sparse=True)
encoded=encoded.sparse.to_coo()
encoded=encoded.tocsr()


Xtrain=encoded[:len(train)]
Xtest=encoded[len(train):]


kf=StratifiedKFold(n_splits=10)


def objective(trial):
    C=trial.suggest_loguniform('C', 10e-10, 10)
    model=LogisticRegression(C=C, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)
    score=-cross_val_score(model, Xtrain, Ytrain, cv=kf, scoring='roc_auc').mean()
    return score
study=optuna.create_study()


#study.optimize(objective, n_trials=50)

#print(study.best_params)

#print(-study.best_value)
#params=study.best_params


model=LogisticRegression(C=0.07536298444122952, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)
model.fit(Xtrain, Ytrain)
predictions=model.predict_proba(Xtest)[:,1]
submission['target']=predictions
submission.to_csv('submit.csv')
submission.head()






