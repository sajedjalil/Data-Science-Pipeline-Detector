# This kernel is forked by Andy Harless kernel "Simple FFNN from Dromosys Features".
# https://www.kaggle.com/aharless/simple-ffnn-from-dromosys-features

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import sklearn.linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc
import os

print(os.listdir("../input"))
print(os.listdir("../input/save-dromosys-features"))

df = pd.read_pickle('../input/save-dromosys-features/df.pkl.gz')
print("Raw shape: ", df.shape)

y = df['TARGET']
feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
X = df[feats]
print("X shape: ", X.shape, "    y shape:", y.shape)

print("\nPreparing data...")
X = X.fillna(X.mean()).clip(-1e11,1e11)
scaler = MinMaxScaler()
scaler.fit(X)
training = y.notnull()
testing = y.isnull()
X_train = scaler.transform(X[training])
X_test = scaler.transform(X[testing])
y_train = np.array(y[training])
print( X_train.shape, X_test.shape, y_train.shape )

folds = KFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros(X_test.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
    trn_x, trn_y = X_train[trn_idx], y_train[trn_idx]
    val_x, val_y = X_train[val_idx], y_train[val_idx]
    
    clf = Ridge(alpha=20, copy_X=True, fit_intercept=True, solver='auto',max_iter=10000,normalize=False, random_state=0,  tol=0.0025)
    
    clf.fit(trn_x, trn_y)
    
    oof_preds[val_idx] = clf.predict(val_x)
    sub_preds += clf.predict(X_test) / folds.n_splits
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()

sub = pd.DataFrame()
sub['SK_ID_CURR'] = df[testing]['SK_ID_CURR']
sub['TARGET'] = sub_preds
# some TARGET are less than 0.
# I changed them forcibly.
# Try another good way and share it.
sub.loc[sub['TARGET'] < 0, 'TARGET'] = 0
sub[['SK_ID_CURR', 'TARGET']].to_csv('sub_ridge_10fold.csv', index= False)

print( sub.head() )

