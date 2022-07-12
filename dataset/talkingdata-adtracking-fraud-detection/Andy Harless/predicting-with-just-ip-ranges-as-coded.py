# As Joe Eddy noted here:
#   https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/51236#292536
# the encoding for IP address is not random

# Ranges of coded IP address (not just specific IP address)
# seem to contain information that is useful for predicting the target.

# But something werid is going on...


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import gc


train_cols = ['ip', 'is_attributed']
test_cols = ['ip', 'click_id']

dtypes = {
        'ip'            : 'uint32',
        'click_id'      : 'uint32'
        }
        
train = pd.read_csv('../input/train.csv', dtype=dtypes, header=0, usecols=train_cols)
test = pd.read_csv('../input/test.csv', dtype=dtypes, header=0, usecols=test_cols)

X_train, X_val, y_train, y_val = train_test_split( train[['ip']], train[['is_attributed']], 
                                                   test_size=0.1, shuffle=False )

X_test = test[['ip']]
submit = test[['click_id']]
del train
del test
gc.collect()

clf = DecisionTreeClassifier(criterion = 'entropy',max_depth = 4)
clf.fit(X_train.values.reshape(-1,1),y_train.values)

val_pred = clf.predict_proba(X_val)[:,1]
y_pred = clf.predict_proba(X_test)[:,1]

# Ignore IP addresses that are directly present in the training set 
training_ips = X_train.ip.unique()

test_old_ip_mask = X_test.ip.isin( training_ips )
test_new_ip_mask = ~test_old_ip_mask
y_median = pd.Series(y_pred[ test_new_ip_mask ]).median()
y_pred[ test_old_ip_mask ] = y_median

val_old_ip_mask = X_val.ip.isin( training_ips )
val_new_ip_mask = ~val_old_ip_mask
val_median = y_val[ val_new_ip_mask ].median()
val_pred[ val_old_ip_mask ] = val_median

print('VALIDATION SCORE')
print( roc_auc_score( np.array(y_val), val_pred )  )

submit['is_attributed'] = y_pred

submit.to_csv('pred_from_just_ip.csv', index=False, float_format='%.7f')