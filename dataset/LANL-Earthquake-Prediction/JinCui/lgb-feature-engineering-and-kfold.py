import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
import gc
from tqdm import tqdm
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

'''
def gen_features(X, n_subchunks=150):
    features = []
    datalist = []
    units = 150000//n_subchunks
    for i in range(n_subchunks):
        datalist.append(X[ units*i : units*i+units ])
    for segment in datalist:
        for i in range(101):
            features.append(np.quantile(segment, i/100))
        for i in range(10):
            if not i%10:
                pass
            else:
                features.append(np.quantile(segment, (990+i)/1000 ))
                features.append(np.quantile(segment, i/1000 ))
        features.append(segment.mean())
        features.append(segment.std())
        features.append(segment.mad())
        features.append(segment.skew())
        features.append(segment.kurtosis())
        features.append(np.sqrt(np.mean(segment**2)))
        features.append(np.abs(segment).mean())
        features.append(np.abs(segment).std())
        for i in range(101):
            features.append( np.quantile(np.abs(np.diff(segment)), (900+i)/1000) )

        features.append(np.abs(np.diff(segment)).min())
        features.append(np.abs(np.diff(segment)).mean())
        features.append(np.abs(np.diff(segment)).std())
        features.append(np.sqrt( np.abs(np.mean((np.diff(segment))**2)) ))
    for i in range(1001):
        features.append(np.quantile(X, i/1000))
        
    features.append(X.mean())
    features.append(X.std())
    features.append(X.mad())
    features.append(X.skew())
    features.append(X.kurtosis())
    features.append(np.sqrt(np.mean(X**2)))
    features.append(np.abs(X).mean())
    features.append(np.abs(X).std())
    for i in range(101):
        features.append( np.quantile(np.abs(np.diff(X)), (900+i)/1000) )
    for i in range(10):
        features.append( np.quantile(np.abs(np.diff(X)), i/10) )
    features.append(np.abs(np.diff(X)).mean())
    features.append(np.abs(np.diff(X)).std())
    features.append(np.sqrt( np.abs(np.mean((np.diff(X))**2)) ))
    return pd.Series(features)
    
    
train = pd.read_csv("../input/LANL-Earthquake-Prediction/train.csv", iterator=True, chunksize=150000, 
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})


X_train = pd.DataFrame()
y_train = pd.Series()
df = pd.DataFrame()

################ Generating features for train data ################ 
i = 0
for chunk in train:
    i += 1
    if not i%1000:
        print(i)  
    # append will append data vertically!
    df = df.append(chunk)
    if len(df) >= 150000:
        df = df[-150000:]
        ch = gen_features(df['acoustic_data'])
        X_train = X_train.append(ch, ignore_index=True)
        y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))
        
################ Generating features for test data ################ 
submission = pd.read_csv("../input/LANL-Earthquake-Prediction/sample_submission.csv", index_col='seg_id')

X_test = pd.DataFrame()
for seg_id in submission.index:
    
    seg = pd.read_csv('../input/LANL-Earthquake-Prediction/test/' + seg_id + '.csv')
    ch = gen_features(seg['acoustic_data'])
    X_test = X_test.append(ch, ignore_index=True)

# X_train.to_hdf('feature_engineering.hdf5', key='X_train')
# X_test.to_hdf('feature_engineering.hdf5', key='X_test')
# y_train.to_hdf('feature_engineering.hdf5', key='y_train')

'''

X_train =pd.read_hdf('../input/lan-earthquake-features/FE_60223.h5', key='X_train')
X_test = pd.read_hdf('../input/lan-earthquake-features/FE_60223.h5', key='X_test')
y_train =pd.read_hdf('../input/lan-earthquake-features/FE_60223.h5', key='y_train')

def gen_quake_idx(y_train):
    diff = y_train.diff()
    quake_pos = np.where(diff>0)[0]
    quake_interval = np.diff(quake_pos)
    quake_interval = np.append(quake_pos[0], quake_interval)
    quake_idx = []
    for i in range(len(quake_interval)):
        quake_idx.append(np.repeat(i, quake_interval[i]))
    quake_idx = np.concatenate(quake_idx)
    quake_idx = np.append(quake_idx, np.repeat(15, len(y_train) - len(quake_idx)))
    return quake_idx

# Quake-based
quake_idx = gen_quake_idx(y_train)
# TTF-based
ttf_idx = y_train.astype('int')
ttf_idx[ttf_idx==16] = 15


def kfold_lightgbm(X_train, y_train, X_test, split_idx=None,
                   n_splits=4, random_state=42, stratified=True, **param):
    
    if stratified:
        assert split_idx is not None
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(folds.split(X_train, split_idx))
    else:
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(folds.split(X_train))
    
    oof_preds = np.zeros(X_train.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in X_train.columns]
    
    sub_preds = np.zeros(X_test.shape[0])
    for n_fold, (train_idx, valid_idx) in enumerate(splits):
        print('Fold_{}'.format(n_fold))
        train_x, train_y = X_train.iloc[train_idx], y_train.iloc[train_idx]
        valid_x, valid_y = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                )
        lgb_test = lgb.Dataset(valid_x,
                              label=valid_y,
                              )
        lgb_param = param
        lgb_param['seed'] = int(2**n_fold)
        lgb_param['drop_seed'] = int(2**n_fold)
        lgb_param['feature_fraction_seed'] = int(2**n_fold),
        
        reg = lgb.train(lgb_param, 
                        lgb_train, 
                        valid_sets=[lgb_train, lgb_test],
                        num_boost_round=40000, 
                        verbose_eval=500, 
                        early_stopping_rounds=2000
                        )
        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        
        sub_preds += reg.predict(X_test, num_iteration=reg.best_iteration) / n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:].index
        sorted_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
        sorted_features = sorted_features[['feature', 'importance']].groupby('feature').sum()
        sorted_features = sorted_features.sort_values(by='importance', ascending=False)
        
        print('Fold_{} MAE is {}'.format(n_fold, mean_absolute_error(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y, train_idx, valid_idx
        gc.collect()
    MAE = mean_absolute_error(y_train, oof_preds)
    print('Final MAE is {}'.format(MAE))
   
    return oof_preds, sub_preds, sorted_features
  

params_0 = {
    'learning_rate': 0.01, 
    'feature_fraction': 0.01, 
    'num_leaves': 80, 
    'min_data_in_leaf': 40, 
    'max_depth': 11, 
    'reg_alpha': 5, 
    'reg_lambda': 20, 
    'subsample': 0.55, 
    'min_child_weight': 46, 
    'min_split_gain': 11, 
    'top_rate': 0.16, 
    'other_rate': 0.17,
    'objective': 'regression',
    'metric': 'l1',
    'verbosity': -1,
    'boosting_type': 'goss',
    'task': 'train'
    }

_, _, sorted_features = kfold_lightgbm(X_train, y_train, X_test, split_idx=quake_idx,**params_0)

## Select most important features

sub_cols = list(sorted_features[:450].index)
print('Selected {} most important features'.format(len(sub_cols)))

params_1 = {
    'learning_rate': 0.0034087259579562296, 
    'feature_fraction': 0.15, 
    'num_leaves': 238, 
    'min_data_in_leaf': 15, 
    'max_depth': 11, 
    'reg_alpha': 47, 
    'reg_lambda': 20, 
    'subsample': 0.5536642092639865, 
    'min_child_weight': 46, 
    'min_split_gain': 11, 
    'top_rate': 0.16274730898229386, 
    'other_rate': 0.17455674665843984,
    'objective': 'regression',
    'metric': 'l1',
    'verbosity': -1,
    'boosting_type': 'goss',
    'task': 'train'
    }
    
oof_preds_0, sub_preds_0, _ = kfold_lightgbm(X_train[sub_cols], y_train, X_test[sub_cols], split_idx=quake_idx,  n_splits=4, **params_1)
oof_preds_1, sub_preds_1, _ = kfold_lightgbm(X_train[sub_cols], y_train, X_test[sub_cols], split_idx=ttf_idx,  n_splits=4, **params_1)
oof_preds_2, sub_preds_2, _ = kfold_lightgbm(X_train[sub_cols], y_train, X_test[sub_cols], stratified=False,  n_splits=4, **params_1)

oof_preds_m = np.vstack([oof_preds_0, oof_preds_1, oof_preds_2]).mean(axis=0)
sub_preds_m = np.vstack([sub_preds_0, sub_preds_1, sub_preds_2]).mean(axis=0)
mean_absolute_error(y_train, oof_preds_m)
print('stacked prediction score: {}'.format(mean_absolute_error(y_train, oof_preds_m)))

folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)


plt.subplots()
plt.figure(figsize=(12,8))
plt.plot(range(len(y_train)), oof_preds_m)
plt.plot(range(len(y_train)), y_train)
plt.title('Comparison of out-of-fold prediction and true target')
plt.savefig('lgb_oof_vs_target_stacked.png')


error_m = np.abs(oof_preds_m - y_train)
plt.subplots()
plt.figure(figsize=(12,8))
plt.scatter(y_train, error_m, s=10)
plt.title('Where the error comes from')
plt.savefig('lgb_oof_error_by_target_stacked.png')



submission_quake_based = pd.read_csv("../input/LANL-Earthquake-Prediction/sample_submission.csv", index_col='seg_id')
submission_quake_based['time_to_failure'] = sub_preds_0
submission_quake_based.to_csv('lgb_quake_based.csv')

submission_ttf_based = pd.read_csv("../input/LANL-Earthquake-Prediction/sample_submission.csv", index_col='seg_id')
submission_ttf_based['time_to_failure'] = sub_preds_1
submission_ttf_based.to_csv('lgb_ttf_based.csv')

submission_KFold = pd.read_csv("../input/LANL-Earthquake-Prediction/sample_submission.csv", index_col='seg_id')
submission_KFold['time_to_failure'] = sub_preds_2
submission_KFold.to_csv('lgb_KFold.csv')

submission_mean = pd.read_csv("../input/LANL-Earthquake-Prediction/sample_submission.csv", index_col='seg_id')
submission_mean['time_to_failure'] = sub_preds_m
submission_mean.to_csv('lgb_mean.csv')

