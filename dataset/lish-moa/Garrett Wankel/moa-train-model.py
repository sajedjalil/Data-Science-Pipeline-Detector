
import pandas as pd 
import numpy as np 

import pickle
import time

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


        
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
tt_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')


kf = KFold(n_splits=10, shuffle=True, random_state=22)
ovr_class = OneVsRestClassifier(SGDClassifier(loss='log', max_iter=25000, n_jobs=-1, random_state=34, learning_rate='constant', eta0=0.002, alpha=0.004, shuffle=True), n_jobs=-1)
clf = Pipeline([('ss', StandardScaler()), ('classifier', ovr_class)])


full_dfs = [tt_scored]
def col_drop(df):
    df = df.drop(columns=['sig_id'], axis=1, inplace=True)
    return df
    
    
for df in full_dfs:
    col_drop(df)

    
dfs = [train_features, test_features]


def cleaner(df):
    df['cp_type'] = df['cp_type'].map({'ctl_vehicle': 0, 'trt_cp': 1})
    df['cp_time'] = df['cp_time'].map({24: 1, 48: 2, 72: 3})
    df['cp_dose'] = df['cp_dose'].map({'D1': 0 , 'D2': 1})
    return df


for df in dfs:
    cleaner(df)
    
    
# keep_idx_test = test_features[test_features.cp_type != 0].index
# keep_idx_train = train_features[train_features.cp_type != 0].index

# test_features = test_features.loc[keep_idx_test] 
# train_features = train_features.loc[keep_idx_train]
# tt_scored = tt_scored.loc[keep_idx_train]


tr_cols = train_features.loc[:, 'cp_type':]
test_cols = test_features.loc[:, 'cp_type':]


col_list = ['g-496', 'g-333', 'g-676', 'g-127', 'g-39', 'g-360', 'g-28', 'g-19', 'g-184', 'g-110', 'g-687', 'g-216',
            'g-15', 'g-626', 'g-393', 'g-667', 'g-164', 'g-688', 'g-754', 'g-557', 'g-363', 'g-132', 'g-435', 'g-536',
            'g-550', 'g-481','g-611', 'g-18', 'g-756', 'g-331', 'g-618', 'g-718', 'g-370', 'g-219','g-153','g-46','g-238',
            'g-23','g-707','g-213','g-307','g-104']
dfs_2 = [tr_cols,test_cols]
 
def outlier_drop(df, col):
    df = df.drop([col], axis=1, inplace=True)
    return df
for col in col_list:
    for df in dfs_2:
        outlier_drop(df, col)

print(tr_cols.shape, test_cols.shape)
X, y, test = np.array(tr_cols), np.array(tt_scored), np.array(test_cols)


oof_preds = np.zeros(y.shape)
oof_losses = []
list_preds = []


for k_f, (tr_idx, t_idx) in enumerate(kf.split(X, y)):
    fold_start = time.time()
    
    X_train, X_val = X[tr_idx], X[t_idx]
    y_train, y_val = y[tr_idx], y[t_idx]
    
    clf1 = clf.fit(X_train, y_train)
    val_preds = clf.predict_proba(X_val)
    val_preds = np.array(val_preds)
    
    oof_preds[t_idx] = np.array(val_preds)
    
    loss = log_loss(np.ravel(y_val), np.ravel(val_preds))
    oof_losses.append(loss)
    
    preds = clf.predict_proba(test)
    list_preds.append(preds)
    
    
    fold_end = time.time()
    print('fold time: ', fold_end - fold_start)


    pickle.dump(clf1, open('OvR', 'wb'))
    model = pickle.load(open('OvR', 'rb'))
    print('model type: ', type(model))
    
    
print(oof_losses)
print(log_loss(np.ravel(y), np.ravel(oof_preds)))    
    