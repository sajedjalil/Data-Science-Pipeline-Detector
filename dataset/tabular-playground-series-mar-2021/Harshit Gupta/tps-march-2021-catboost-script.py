import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import os
import joblib
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import catboost
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge



train = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')
sample = pd.read_csv('../input/tabular-playground-series-mar-2021/sample_submission.csv')
print(train.shape, test.shape, sample.shape)


cat_cols = train.select_dtypes(include=['object']).columns
cont_cols = train.select_dtypes(include=['number']).drop(['id', 'target'], 1).columns


cv = StratifiedKFold(n_splits=5, shuffle=True)
FEATURES = cat_cols.append(cont_cols)

X = train[FEATURES]
y = train['target']

NUM_BOOST_ROUNDS = 10000
EARLY_STOPPING_ROUNDS = 200

oof_df = train[['id', 'target']].copy()
fold = 1

for train_idx, val_idx in cv.split(X, y):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = catboost.CatBoostClassifier(num_boost_round=NUM_BOOST_ROUNDS, silent=True)
    model.fit(X_train, y_train, cat_cols, eval_set=(X_val, y_val), early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    
    preds = model.predict_proba(X_val)
    test_preds = model.predict_proba(test[FEATURES])
    train_preds = model.predict_proba(train[FEATURES])
    oof_df.loc[val_idx, 'oof'] = preds[:, 1]
    sample[f'fold_{fold}'] = test_preds[:, 1]
    oof_df[f'fold_{fold}'] = train_preds[:, 1]
    print(roc_auc_score(oof_df.loc[val_idx, 'target'], preds[:, 1]))
    joblib.dump(model, f'model_{fold}')
    fold += 1

oof_df
meta_features = ['fold_1','fold_2','fold_3','fold_4','fold_5']
meta_model = Ridge()

X_train, X_val, y_train, y_val = train_test_split(oof_df[meta_features], train['target'], test_size=0.3, shuffle=True)
meta_model.fit(X_train, y_train)

k = meta_model.predict(X_val)
print(roc_auc_score(y_val, k))

sample['target'] = (sample[meta_features].mean(axis=1) + meta_model.predict(sample[meta_features]))/2
sample[['id', 'target']].to_csv('submission.csv', index=False)


