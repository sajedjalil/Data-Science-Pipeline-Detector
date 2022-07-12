# %% [code]
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb


# %% [code]
train_df = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


train_df.drop(['ID_code'],axis=1,inplace=True)

test_id_list = test_df.ID_code
test_df.drop(['ID_code'],axis=1,inplace=True)


# %% [code]
X = train_df.drop('target',axis=1)
y = train_df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify = y, random_state=42)

param_dict = {
    'max_depth': range(1,10),
    'gamma': np.arange(0,0.5,0.05),
    'lambda': np.geomspace(1, 5, num=10),
    'min_child_weight': [1, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.5,0.6, 0.8, 1.0]
}
xgb = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric= 'auc')

rscv = RandomizedSearchCV(xgb, param_dict,n_iter=800, scoring = 'roc_auc', n_jobs = -1, verbose = 1 , cv= StratifiedKFold(4).split(X_train,y_train))
rscv.fit(X_train,y_train)

print("Best Paramaeter = ", rscv.best_estimator_ )

# %% [code]
print("train accuracy score = ", accuracy_score(y_train,rscv.predict(X_train)))
print("test accuracy score = ", accuracy_score(y_test,rscv.predict(X_test)))
print("ROC_AUC score = ", roc_auc_score(y_test,rscv.predict_proba(X_test)[:,1] ))


# submitting output
output_submission = pd.DataFrame(zip(test_id_list,rscv.predict_proba(test_df)[:,1]), columns = ['ID_code','target'])
output_submission.to_csv('/kaggle/working/output_submission.csv',index=False)