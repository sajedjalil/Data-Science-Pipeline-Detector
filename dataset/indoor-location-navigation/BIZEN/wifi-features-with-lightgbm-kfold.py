# ------------------------------------------------------------------------------
# Import libraries
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path
import glob

from sklearn.model_selection import KFold
import lightgbm as lgb

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
N_SPLITS = 20
SEED = 42

# ------------------------------------------------------------------------------
# File path definition
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


def score_log(df: pd.DataFrame, num_files: int, nam_file: str, data_shape: tuple, n_fold: int, seed: int, mpe: float):
    score_dict = {'n_files': num_files, 'file_name': nam_file, 'shape': data_shape, 'fold': n_fold, 'seed': seed, 'score': mpe}
    # noinspection PyTypeChecker
    df = pd.concat([df, pd.DataFrame.from_dict([score_dict])])
    df.to_csv(LOG_PATH / f"log_score.csv", index=False)
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
oof = list()
predictions = list()
for n_files, file in enumerate(train_files):
    data = pd.read_csv(file, index_col=0)
    test_data = pd.read_csv(test_files[n_files], index_col=0)

    oof_x, oof_y, oof_f = np.zeros(data.shape[0]), np.zeros(data.shape[0]), np.zeros(data.shape[0])
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

        modelx = lgb.LGBMRegressor(**lgb_params)
        with timer("fit X"):
            modelx.fit(X_train, y_trainx,
                       eval_set=[(X_valid, y_validx)],
                       eval_metric='rmse',
                       verbose=False,
                       early_stopping_rounds=20
                       )

        modely = lgb.LGBMRegressor(**lgb_params)
        with timer("fit Y"):
            modely.fit(X_train, y_trainy,
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

        oof_x[val_idx] = modelx.predict(X_valid)
        oof_y[val_idx] = modely.predict(X_valid)
        oof_f[val_idx] = modelf.predict(X_valid).astype(int)

        preds_x += modelx.predict(test_data.iloc[:, :-1]) / N_SPLITS
        preds_y += modely.predict(test_data.iloc[:, :-1]) / N_SPLITS
        preds_f_arr[:, fold] = modelf.predict(test_data.iloc[:, :-1]).astype(int)

        score = comp_metric(oof_x[val_idx], oof_y[val_idx], oof_f[val_idx],
                            y_validx.to_numpy(), y_validy.to_numpy(), y_validf.to_numpy())
        print(f"fold {fold}: mean position error {score}")
        score_df = score_log(score_df, n_files, os.path.basename(file), data.shape, fold, SEED, score)

    print("*+"*40)
    print(f"file #{n_files}, shape={data.shape}, name={os.path.basename(file)}")
    score = comp_metric(oof_x, oof_y, oof_f,
                        data.iloc[:, -4].to_numpy(), data.iloc[:, -3].to_numpy(), data.iloc[:, -2].to_numpy())
    oof.append(score)
    print(f"mean position error {score}")
    print("*+"*40)
    score_df = score_log(score_df, n_files, os.path.basename(file), data.shape, 999, SEED, score)

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