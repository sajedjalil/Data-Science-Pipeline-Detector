# IP addresses that haven't appeared before are more likely to download

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

training_ips = X_train.ip.unique()
test_new = ~X_test.ip.isin( training_ips )
val_new = ~X_val.ip.isin( training_ips )

val_pred = 0*val_new
val_pred[val_new] = .5

print('VALIDATION SCORE')
print( roc_auc_score( np.array(y_val), val_pred )  )

y_pred = 0*test_new
y_pred[test_new] = .5


submit['is_attributed'] = y_pred

submit.to_csv('pred_from_just_ip.csv', index=False, float_format='%.2f')