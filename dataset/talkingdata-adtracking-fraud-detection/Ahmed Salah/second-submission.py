import sys
import wordbatch
import threading
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
import numpy as np
import gc
from contextlib import contextmanager
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error

df=pd.read_csv("../input/train.csv",nrows=20000000)
del df['attributed_time']
new=[]
for i in (df['click_time']):
    element=""
    i=i.replace(":","11")
    i=i.replace(" ","12")
    i=i.replace("-","13")
    new.append(i)
df2=pd.DataFrame({'click_time':new})
df['click_time']=df2
print("here")
Y_train=df['is_attributed'].values
del df['is_attributed']
X_train=df.values
print(1)
df2=pd.read_csv('../input/train.csv')
print(2)
X_test=df2.values
print(3)
lgb_train = lgb.Dataset(X_train, Y_train)
print(4)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
params = {
    'learning_rate': 0.1,
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99 
}
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
df_sub = pd.DataFrame()
df_sub['click_id']=df2['click_id'].astype('int')
df_sub['is_attributed']=y_pred
df_sub.to_csv("my_file.csv", index=False,float_format='%.10f')