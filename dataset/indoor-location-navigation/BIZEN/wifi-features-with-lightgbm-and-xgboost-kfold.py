# ------------------------------------------------------------------------------
# Import libraries
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path
import glob
import os

from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb

import psutil
import random
import os
import time
import sys
import math
from contextlib import contextmanager

# ------------------------------------------------------------------------------
# Fixed values
# ------------------------------------------------------------------------------
N_SPLITS = 5
SEED = 42

# ------------------------------------------------------------------------------
# File and model path definition
# ------------------------------------------------------------------------------
LOG_PATH = Path("./log/")
LOG_PATH.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
@contextmanager
def timer(name: str):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    try:
        yield
    finally:
        m1 = p.memory_info()[0] / 2. ** 30
        delta = m1 - m0
        sign = '+' if delta >= 0 else '-'
        delta = math.fabs(delta)
        print(f"[{m1:.1f}GB({sign}{delta:.1f}GB): {time.time() - t0:.3f}sec] {name}", file=sys.stderr)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    
def comp_metric(xhat, yhat, fhat, x, y, f):
    intermediate = np.sqrt(np.power(xhat-x, 2) + np.power(yhat-y, 2)) + 15 * np.abs(fhat-f)
    return intermediate.sum()/xhat.shape[0]


def score_log(df: pd.DataFrame, num_files: int, nam_file: str, data_shape: tuple, n_fold: int, seed: int, model: str, mpe: float):
    score_dict = {'n_files': num_files, 'file_name': nam_file, 'shape': data_shape, 'fold': n_fold, 'seed': seed, 'model': model, 'score': mpe}
    # noinspection PyTypeChecker
    df = pd.concat([df, pd.DataFrame.from_dict([score_dict])])
    df.to_csv(LOG_PATH / "log_score.csv", index=False)
    return df


# ------------------------------------------------------------------------------
# Set seed
# ------------------------------------------------------------------------------
set_seed(SEED)

# ------------------------------------------------------------------------------
# Read data
# ------------------------------------------------------------------------------
feature_dir = "../input/indoor-navigation-and-location-wifi-features"
train_files = sorted(glob.glob(os.path.join(feature_dir, '*_train.csv')))
test_files = sorted(glob.glob(os.path.join(feature_dir, '*_test.csv')))
subm = pd.read_csv('../input/indoor-location-navigation/sample_submission.csv', index_col=0)

# ------------------------------------------------------------------------------
# Define parameters for models
# ------------------------------------------------------------------------------
lgb_params = {'objective': 'root_mean_squared_error',
              'boosting_type': 'gbdt',
              'n_estimators': 50000,
              'learning_rate': 0.1,
              'num_leaves': 90,
              'colsample_bytree': 0.4,
              'subsample': 0.6,
              'subsample_freq': 2,
              'bagging_seed': SEED,
              'reg_alpha': 8,
              'reg_lambda': 2,
              'random_state': SEED,
              'n_jobs': -1
              }

xgb_params = {'objective': 'reg:squarederror',
              'booster': 'gbtree',
              'eval_metric': 'rmse',
              'n_estimators': 50000,
              'learning_rate': 0.1,
              'max_depth': 8,
              'colsample_bytree': 0.4,
              'subsample': 0.6,
              'alpha': 8,
              'lambda': 2,
              'random_state': SEED,
              'tree_method': 'gpu_hist'
              }

lgb_f_params = {'objective': 'multiclass',
                'boosting_type': 'gbdt',
                'n_estimators': 50000,
                'learning_rate': 0.1,
                'num_leaves': 90,
                'colsample_bytree': 0.4,
                'subsample': 0.6,
                'subsample_freq': 2,
                'bagging_seed': SEED,
                'reg_alpha': 10,
                'reg_lambda': 2,
                'random_state': SEED,
                'n_jobs': -1
                }

# ------------------------------------------------------------------------------
# Training and inference
# ------------------------------------------------------------------------------
score_df = pd.DataFrame()
lgb_oof, xgb_oof = list(), list()
predictions = list()
for n_files, file in enumerate(train_files):
    data = pd.read_csv(file, index_col=0)
    test_data = pd.read_csv(test_files[n_files], index_col=0)

    lgb_oof_x, lgb_oof_y = np.zeros(data.shape[0]), np.zeros(data.shape[0])
    xgb_oof_x, xgb_oof_y = np.zeros(data.shape[0]), np.zeros(data.shape[0])
    oof_f = np.zeros(data.shape[0])
    preds_x, preds_y = 0, 0
    preds_f_arr = np.zeros((test_data.shape[0], N_SPLITS))

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (trn_idx, val_idx) in enumerate(kf.split(data.iloc[:, :-4])):
        X_train = data.iloc[trn_idx, :-4]
        y_trainx = data.iloc[trn_idx, -4]
        y_trainy = data.iloc[trn_idx, -3]
        y_trainf = data.iloc[trn_idx, -2]

        X_valid = data.iloc[val_idx, :-4]
        y_validx = data.iloc[val_idx, -4]
        y_validy = data.iloc[val_idx, -3]
        y_validf = data.iloc[val_idx, -2]

        lgb_modelx = lgb.LGBMRegressor(**lgb_params)
        xgb_modelx = xgb.XGBRegressor(**xgb_params)
        with timer("fit X"):
            lgb_modelx.fit(X_train, y_trainx,
                           eval_set=[(X_valid, y_validx)],
                           eval_metric='rmse',
                           verbose=False,
                           early_stopping_rounds=20
                           )
            xgb_modelx.fit(X_train, y_trainx,
                           eval_set=[(X_valid, y_validx)],
                           eval_metric='rmse',
                           verbose=False,
                           early_stopping_rounds=20
                           )

        lgb_modely = lgb.LGBMRegressor(**lgb_params)
        xgb_modely = xgb.XGBRegressor(**xgb_params)
        with timer("fit Y"):
            lgb_modely.fit(X_train, y_trainy,
                           eval_set=[(X_valid, y_validy)],
                           eval_metric='rmse',
                           verbose=False,
                           early_stopping_rounds=20
                           )
            xgb_modely.fit(X_train, y_trainy,
                           eval_set=[(X_valid, y_validy)],
                           eval_metric='rmse',
                           verbose=False,
                           early_stopping_rounds=20
                           )

        modelf = lgb.LGBMClassifier(**lgb_f_params)
        with timer("fit F"):
            modelf.fit(X_train, y_trainf,
                       eval_set=[(X_valid, y_validf)],
                       eval_metric='multi_logloss',
                       verbose=False,
                       early_stopping_rounds=20
                       )

        lgb_oof_x[val_idx] = lgb_modelx.predict(X_valid)
        lgb_oof_y[val_idx] = lgb_modely.predict(X_valid)
        xgb_oof_x[val_idx] = xgb_modelx.predict(X_valid)
        xgb_oof_y[val_idx] = xgb_modely.predict(X_valid)
        oof_f[val_idx] = modelf.predict(X_valid).astype(int)

        preds_x += lgb_modelx.predict(test_data.iloc[:, :-1]) / (N_SPLITS*2)
        preds_y += lgb_modely.predict(test_data.iloc[:, :-1]) / (N_SPLITS*2)
        preds_x += xgb_modelx.predict(test_data.iloc[:, :-1]) / (N_SPLITS*2)
        preds_y += xgb_modely.predict(test_data.iloc[:, :-1]) / (N_SPLITS*2)
        preds_f_arr[:, fold] = modelf.predict(test_data.iloc[:, :-1]).astype(int)

        lgb_score = comp_metric(lgb_oof_x[val_idx], lgb_oof_y[val_idx], oof_f[val_idx],
                                y_validx.to_numpy(), y_validy.to_numpy(), y_validf.to_numpy())
        xgb_score = comp_metric(xgb_oof_x[val_idx], xgb_oof_y[val_idx], oof_f[val_idx],
                                y_validx.to_numpy(), y_validy.to_numpy(), y_validf.to_numpy())
        print(f"fold {fold}: mean position error: lgb {lgb_score}, xgb {xgb_score}")
        score_df = score_log(score_df, n_files, os.path.basename(file), data.shape, fold, SEED, 'lgb', lgb_score)
        score_df = score_log(score_df, n_files, os.path.basename(file), data.shape, fold, SEED, 'xgb', xgb_score)

    print("*+"*40)
    print(f"file #{n_files}, shape={data.shape}, name={os.path.basename(file)}")
    lgb_score = comp_metric(lgb_oof_x, lgb_oof_y, oof_f,
                            data.iloc[:, -4].to_numpy(), data.iloc[:, -3].to_numpy(), data.iloc[:, -2].to_numpy())
    xgb_score = comp_metric(xgb_oof_x, xgb_oof_y, oof_f,
                            data.iloc[:, -4].to_numpy(), data.iloc[:, -3].to_numpy(), data.iloc[:, -2].to_numpy())
    lgb_oof.append(lgb_score)
    xgb_oof.append(xgb_score)
    print(f"mean position error: lgb {lgb_score}, xgb {xgb_score}")
    print("*+"*40)
    score_df = score_log(score_df, n_files, os.path.basename(file), data.shape, 999, SEED, 'lgb', lgb_score)
    score_df = score_log(score_df, n_files, os.path.basename(file), data.shape, 999, SEED, 'xgb', xgb_score)

    preds_f_mode = stats.mode(preds_f_arr, axis=1)
    preds_f = preds_f_mode[0].astype(int).reshape(-1)
    test_preds = pd.DataFrame(np.stack((preds_f, preds_x, preds_y))).T
    test_preds.columns = subm.columns
    test_preds.index = test_data["site_path_timestamp"]
    test_preds["floor"] = test_preds["floor"].astype(int)
    predictions.append(test_preds)

# ------------------------------------------------------------------------------
# Submit the result
# ------------------------------------------------------------------------------
all_preds = pd.concat(predictions)
all_preds = all_preds.reindex(subm.index)
all_preds.to_csv('submission.csv')
