# %% [code]
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

import xgboost as xgb

# %% [code]
train_df = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

# %% [code]
train_df.drop(['ID_code'],axis=1,inplace=True)

test_id_list = test_df.ID_code
test_df.drop(['ID_code'],axis=1,inplace=True)

# %% [code]
X = train_df.drop('target',axis=1)
y = train_df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=42)

clf = xgb.XGBClassifier(tree_method='gpu_hist')
                        
xgb_model = clf.fit(X_train,y_train)

print("train accuracy score = ", accuracy_score(y_train,xgb_model.predict(X_train)))
print("test accuracy score = ", accuracy_score(y_test,xgb_model.predict(X_test)))
print("ROC_AUC score = ", roc_auc_score(y_test,xgb_model.predict_proba(X_test)[:,1] ))

# %% [code]
# %% [code]
#submitting output
output_submission = pd.DataFrame(zip(test_id_list,xgb_model.predict_proba(test_df)[:,1]), columns = ['ID_code','target'])
output_submission.to_csv('/kaggle/working/output_submission.csv',index=False)