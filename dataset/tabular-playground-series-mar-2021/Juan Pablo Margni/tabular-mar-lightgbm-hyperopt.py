# %% [code]
import warnings
import joblib
import lightgbm as lgb
from functools import partial
import numpy as np
import pandas as pd
from hyperopt import fmin, hp, STATUS_OK, tpe, Trials
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

np.random.seed(1)

def label_encoder(c):
    lc = LabelEncoder()
    return lc.fit_transform(c)

def preprocess(df):
    dataframe = df.copy()
    categorical_cols = [c for c in dataframe.columns if 'cat' in c]
    numerical_cols = [c for c in dataframe.columns if 'cat' not in c]
    # categorical_df = dataframe[categorical_cols].apply(label_encoder)
    # Onehot encoding for all categorical features
    categorical_df = pd.get_dummies(df[categorical_cols])
    numerical_df = dataframe[numerical_cols]
    dataframe = pd.concat([numerical_df, categorical_df], axis=1)
    return dataframe.drop(columns=['id'])


def train_and_evaluate(params,
                       eval_metric,
                       target_name,
                       all_data,
                       test_frac
                       ):
    train_df, test_df = train_test_split(all_data.copy(), test_size=test_frac, stratify=all_data[target_name])

    y = train_df.pop(target_name).values
    X = train_df

    y_ = test_df.pop(target_name).values
    X_ = test_df

    clf = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    clf.fit(X, y, eval_set=[(X, y), (X_, y_)], eval_metric=eval_metric, early_stopping_rounds=50, verbose=-1)
    best_score = roc_auc_score(y_, clf.predict_proba(X_, num_iteration=clf.best_iteration_)[:, 1])
    print(f'Best validation accuracy: {best_score}')
    return {'loss': -1 * best_score, 'status': STATUS_OK}

# Use trials to save partial results. Save every trials_step
def main(csv_path, target_name, test_frac=0.25, trials_step=50):
    max_trials = trials_step  # initial max_trials. put something small to not have to wait

    all_data = pd.read_csv(csv_path)
    all_data = preprocess(all_data)

    try:  # try to load an already saved trials object, and increase the max
        trials = joblib.load("tabular_mar_suggest.hyperopt")
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    params = {}  # initialize parameters
    params['num_leaves'] = hp.choice('num_leaves', range(20, 300))
    params['min_data_in_leaf'] = hp.choice('min_data_in_leaf', range(10, 100, 10))
    params['max_depth'] = hp.choice('max_depth', range(5, 200))
    params['max_bin'] = hp.choice('max_bin', range(10, 300, 10))
    params['learning_rate'] = hp.uniform('learning_rate', 0.0, 1.0)
    params['sub_feature'] = hp.uniform('sub_feature', 0.0, 1.0)
    params['n_estimators'] = hp.choice('n_estimators', [10000])
    params['lambda'] = hp.choice('lambda', [0.01, 0.001, 0.0001, 0.00001])
    params['metric'] = hp.choice('metric', ['auc'])
    params['objective'] = hp.choice('objective', ['binary'])
    # params['extra_trees'] = hp.choice('extra_trees', [True])
    # params['is_unbalance'] = hp.choice('is_unbalance', [True])
    # params['save_binary'] = hp.choice('save_binary', [True])

    opt_fn = partial(
        train_and_evaluate,
        eval_metric='auc',
        target_name=target_name,
        all_data=all_data,
        test_frac=test_frac
    )

    # Now opt_fn is a function of a single variable, params
    best = fmin(
        opt_fn,
        params,
        algo=tpe.suggest,
        max_evals=max_trials,
        trials=trials
    )

    joblib.dump(trials, "tabular_mar_suggest.hyperopt")
    print(f'Best parameter setting: {best}')


if __name__ == "__main__":
    # Looping and saving 50 times for 100 trials each -> 5000 param combinations
    for i in range(100):
        main('../input/tabular-playground-series-mar-2021/train.csv', 'target')
