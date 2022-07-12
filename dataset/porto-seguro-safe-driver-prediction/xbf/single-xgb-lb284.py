# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import os
import xgboost as xgb
import gc

def read_data():
    root_dir = '../input/'
    df_train = pd.read_csv(os.path.join(root_dir, 'train.csv'), na_values=-1)
    df_train.drop([149161], axis=0, inplace=True)
    df_test = pd.read_csv(os.path.join(root_dir, 'test.csv'), na_values=-1)
    df_y = df_train['target']
    train_id = df_train['id']
    df_train.drop(['id', 'target'], axis=1, inplace=True)
    df_sub = df_test['id'].to_frame()
    df_sub['target'] = 0.0
    df_test.drop(['id'], axis=1, inplace=True)
    return df_train, df_y, df_test, df_sub, train_id

def write_data(df_sub, train_id, stacker_train, sub_filename, train_filename):
    df_sub.to_csv(sub_filename, index=False)
    s_train = pd.DataFrame()
    s_train['id'] = train_id
    s_train['prob'] = stacker_train
    s_train.to_csv(train_filename, index=False)

class SingleXGB(object):
    def __init__(self, X, y, test, skf, N):
        self.X = X
        self.y = y
        self.test = test
        self.skf = skf
        self.N = N

    def oof(self, params, best_rounds, sub, do_logit=True):
        stacker_train = np.zeros((self.X.shape[0], 1))
        dtest = xgb.DMatrix(data=self.test.values)
        for index, (trn_idx, val_idx) in enumerate(self.skf.split(self.X, self.y)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            dtrn = xgb.DMatrix(data=trn_x, label=trn_y)
            dval = xgb.DMatrix(data=val_x, label=val_y)
            print('Train model in fold {0}'.format(index))
            cv_model = xgb.train(
                params=params,
                dtrain=dtrn,
                num_boost_round=best_rounds,
                verbose_eval=10,
            )
            print('Predict in fold {0}'.format(index))
            prob = cv_model.predict(dtest, ntree_limit=best_rounds)
            stacker_train[val_idx,0] = cv_model.predict(dval, ntree_limit=best_rounds)
            sub['target'] += prob / self.N
        if do_logit:
            sub['target'] = 1 / (1 + np.exp(-sub['target']))
            stacker_train = 1 / (1 + np.exp(-stacker_train))
        print('{0} of folds'.format(self.N))
        print('Oof by single xgboost model Done')
        return sub, stacker_train

class Compose(object):
    def __init__(self, transforms_params):
        self.transforms_params = transforms_params
    def __call__(self, df):
        for transform_param in self.transforms_params:
            transform, param = transform_param[0], transform_param[1]
            df = transform(df, **param)
        return df

class Processer(object):
    @staticmethod
    def drop_columns(df, col_names):
        print('Before drop columns {0}'.format(df.shape))
        df = df.drop(col_names, axis=1)
        print('After drop columns {0}'.format(df.shape))
        return df

    @staticmethod
    def dtype_transform(df):
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype(np.int8)
        return df

    @staticmethod
    def negative_one_vals(df):
        df['negative_one_vals'] = MinMaxScaler().fit_transform(df.isnull().sum(axis=1).values.reshape(-1,1))
        return df

    @staticmethod
    def ohe(df_train, df_test, cat_features, threshold=50):
        # pay attention train & test should get_dummies together
        print('Before ohe : train {0}, test {1}'.format(df_train.shape, df_test.shape))
        combine = pd.concat([df_train, df_test], axis=0)
        for column in cat_features:
            temp = pd.get_dummies(pd.Series(combine[column]), prefix=column)
            _abort_cols = []
            for c in temp.columns:
                if temp[c].sum() < threshold:
                    print('column {0} unique value {1} less than threshold {2}'.format(c, temp[c].sum(), threshold))
                    _abort_cols.append(c)
            print('Abort cat columns : {0}'.format(_abort_cols))
            _remain_cols = [ c for c in temp.columns if c not in _abort_cols ]
            # check category number
            combine = pd.concat([combine, temp[_remain_cols]], axis=1)
            combine = combine.drop([column], axis=1)
        train = combine[:df_train.shape[0]]
        test = combine[df_train.shape[0]:]
        print('After ohe : train {0}, test {1}'.format(train.shape, test.shape))
        return train, test


Number_of_folds = 5
comm_skf = StratifiedKFold(n_splits=Number_of_folds, shuffle=True, random_state=2017)


df_train, df_y, df_test, df_sub, train_id = read_data()
skf = comm_skf

## Processing and Feature Engineering
transformer_one = [
    (Processer.drop_columns, dict(col_names=df_train.columns[df_train.columns.str.startswith('ps_calc_')])),
    (Processer.drop_columns, dict(col_names=['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'])),
    (Processer.negative_one_vals, dict()),
    (Processer.dtype_transform, dict()),
]
# execute transforms pipeline
print('Transform train data')
df_train = Compose(transformer_one)(df_train)
print('Transform test data')
df_test = Compose(transformer_one)(df_test)
# execute ohe
df_train, df_test = Processer.ohe(df_train, df_test, [a for a in df_train.columns if a.endswith('cat')])

# extract feature and label for train
X = df_train.values
y = df_y.values

gc.collect()

## cv and oof
params_for_submit = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.04,
    'max_depth': 5,
    'min_child_weight': 9.15,
    'gamma': 0.59,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 10.4,
    'lambda': 5,
    'seed': 2017,
    'nthread': 5,
    'silent': 1,
}
single_xgb = SingleXGB(X=X, y=y, test=df_test, skf=skf, N=Number_of_folds)
best_rounds = 431
df_sub, stacker_train = single_xgb.oof(
    params=params_for_submit,
    best_rounds=best_rounds,
    sub=df_sub,
    do_logit=False
)
# write submit file and train file to local disk
write_data(
    df_sub=df_sub,
    train_id=train_id,
    stacker_train=stacker_train,
    sub_filename='sub_single_xgb_001_test.csv',
    train_filename='single_xgb_001_train.csv'
)
print('Single XGBoost done')