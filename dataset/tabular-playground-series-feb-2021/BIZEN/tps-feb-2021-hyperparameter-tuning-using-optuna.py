# ------------------------------------------------------------------------------
# Import libraries
# ------------------------------------------------------------------------------
import os
import sys
import math
import time
import pickle
import psutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import warnings

import optuna
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
N_FOLDS = 10
N_ESTIMATORS = 30000
SEED = 2021
BAGGING_SEED = 48

N_TRIALS = 50

# ------------------------------------------------------------------------------
# Path definition
# ------------------------------------------------------------------------------
DATA_PATH = Path("../input/tabular-playground-series-feb-2021")
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


def score_log(df: pd.DataFrame, seed: int, num_fold: int, model_name: str, cv: float):
    score_dict = {'date': datetime.now(), 'seed': seed, 'fold': num_fold, 'model': model_name, 'cv': cv}
    # noinspection PyTypeChecker
    df = pd.concat([df, pd.DataFrame.from_dict([score_dict])])
    df.to_csv(LOG_PATH / f"model_score_{model_name}.csv", index=False)
    return df


# ------------------------------------------------------------------------------
# Read data
# ------------------------------------------------------------------------------
with timer("Read data"):
    train_df = pd.read_csv(DATA_PATH / "train.csv")
    test_df = pd.read_csv(DATA_PATH / "test.csv")
    sub_df = pd.read_csv(DATA_PATH / "sample_submission.csv")

# ------------------------------------------------------------------------------
# Data preprocessing
# ------------------------------------------------------------------------------
cont_features = [f'cont{i}' for i in range(14)]
cat_features = [f'cat{i}' for i in range(10)]
all_features = cat_features + cont_features
target_feature = 'target'

target = train_df[target_feature]
train_df = train_df[all_features]
test_df = test_df[all_features]

# Labell encoding
with timer("Label encoding"):
    le = LabelEncoder()
    for col in cat_features:
        le.fit(train_df[col])
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

# RankGauss transformation
with timer("RankGauss transformation"):
    transformer = QuantileTransformer(n_quantiles=100, random_state=2021, output_distribution='normal')
    transformer.fit(train_df[cont_features])
    train_df[cont_features] = transformer.transform(train_df[cont_features])
    test_df[cont_features] = transformer.transform(test_df[cont_features])

# ------------------------------------------------------------------------------
# Optuna: objective()
# ------------------------------------------------------------------------------
def objective(trial, X=train_df, y=target):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=SEED)
    obj_params = {'random_state': SEED,
                  'metric': 'rmse',
                  'n_estimators': N_ESTIMATORS,
                  'n_jobs': -1,
                  'cat_feature': [x for x in range(len(cat_features))],
                  'bagging_seed': SEED,
                  'feature_fraction_seed': SEED,
                  'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
                  'max_depth': trial.suggest_int('max_depth', 6, 127),
                  'num_leaves': trial.suggest_int('num_leaves', 31, 128),
                  'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
                  'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
                  'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.9),
                  'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
                  'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
                  'subsample': trial.suggest_float('subsample', 0.3, 0.9),
                  'max_bin': trial.suggest_int('max_bin', 128, 1024),
                  'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
                  'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
                  'cat_l2': trial.suggest_int('cat_l2', 1, 20)
                  }

    obj_model = LGBMRegressor(**obj_params)
    obj_model.fit(train_x, train_y, eval_set=(test_x, test_y), early_stopping_rounds=100, verbose=False)
    obj_preds = obj_model.predict(test_x, num_iteration=obj_model.best_iteration_)
    obj_rmse = mean_squared_error(test_y, obj_preds, squared=False)
    return obj_rmse


# ------------------------------------------------------------------------------
# Optuna: optimization
# ------------------------------------------------------------------------------
study = optuna.create_study(study_name=f"optimization", direction='minimize')
study.optimize(objective, n_trials=N_TRIALS)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# ------------------------------------------------------------------------------
# Optuna: check history
# ------------------------------------------------------------------------------
print(study.trials_dataframe())
study.trials_dataframe().to_csv(LOG_PATH / "trial_parameters.csv", index=False)
with open(LOG_PATH / 'study.pickle', 'wb') as f:
    pickle.dump(study, f)

# ------------------------------------------------------------------------------
# LightGBM: training and inference
# ------------------------------------------------------------------------------
params = study.best_params
params['random_state'] = SEED
params['metric'] = 'rmse'
params['n_estimators'] = N_ESTIMATORS
params['n_jobs'] = -1
params['cat_feature'] = [x for x in range(len(cat_features))]
params['bagging_seed'] = SEED
params['feature_fraction_seed'] = SEED
with open(LOG_PATH / 'params.pickle', 'wb') as f:
    pickle.dump(params, f)

oof = np.zeros(train_df.shape[0])
preds = 0
score_df = pd.DataFrame()
feature_importances = pd.DataFrame()

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X=train_df)):
    X_train, X_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]
    y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]

    with timer(f"fold {fold}: fit"):
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric='rmse',
                  early_stopping_rounds=100,
                  verbose=0)

    fi_tmp = pd.DataFrame()
    fi_tmp['feature'] = model.feature_name_
    fi_tmp['importance'] = model.feature_importances_
    fi_tmp['fold'] = fold
    fi_tmp['seed'] = SEED
    feature_importances = feature_importances.append(fi_tmp)

    oof[valid_idx] = model.predict(X_valid)
    preds += model.predict(test_df)/N_FOLDS
    rmse = mean_squared_error(y_valid, oof[valid_idx], squared=False)
    score_df = score_log(score_df, SEED, fold, 'lgb', rmse)
    print(f"rmse {rmse}")

rmse = mean_squared_error(target, oof, squared=False)
score_df = score_log(score_df, SEED, 999, 'lgb', rmse)
print("+-"*40)
print(f"rmse {rmse}")

# ------------------------------------------------------------------------------
# Check feature importance
# ------------------------------------------------------------------------------
order = list(feature_importances.groupby('feature').mean().sort_values('importance', ascending=False).index)
fig = plt.figure(figsize=(10, 10))
sns.barplot(x="importance", y="feature", data=feature_importances, order=order)
plt.title("LightGBM importance")
plt.tight_layout()
fig.savefig(LOG_PATH / "feature_importance_lgb.png")

# ------------------------------------------------------------------------------
# Submission
# ------------------------------------------------------------------------------
sub_df.target = preds
sub_df.to_csv(f"submission_cv{rmse:.6f}.csv", index=False)
sub_df.head()

np.save(LOG_PATH / 'train_oof', oof)
np.save(LOG_PATH / 'test_preds', preds)
