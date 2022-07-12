# Stacking Starter based on Allstate Faron's Script
#https://www.kaggle.com/mmueller/allstate-claims-severity/stacking-starter/run/390867
# Preprocessing from Oleg
#https://www.kaggle.com/opanichev/lightgbm-regressor

import pandas as pd
import numpy as np
from scipy.stats import skew
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt


TARGET = 'trip_duration'
NFOLDS = 3
SEED = 0
NROWS = None
SUBMISSION_FILE = '../input/sample_submission.csv'


## Load the data ##
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

ntrain = train.shape[0]
ntest = test.shape[0]


def extract_features(df):
    df['distance'] = np.sqrt(np.power(df['dropoff_longitude'] - df['pickup_longitude'], 2) + np.power(df['dropoff_latitude'] - df['pickup_latitude'], 2))
    df['month'] = df['pickup_datetime'].apply(lambda x: int(x.split(' ')[0].split('-')[1]))
    df['day'] = df['pickup_datetime'].apply(lambda x: int(x.split(' ')[0].split('-')[2]))
    df['hour'] = df['pickup_datetime'].apply(lambda x: int(x.split(' ')[1].split(':')[0]))
    df['minutes'] = df['pickup_datetime'].apply(lambda x: int(x.split(' ')[1].split(':')[1]))
    df['is_weekend'] = ((df.pickup_datetime.astype('datetime64[ns]').dt.dayofweek) // 4 == 1).astype(float)
    df['weekday'] = df.pickup_datetime.astype('datetime64[ns]').dt.dayofweek
    df['is_holyday'] = df.apply(lambda row: 1 if (row['month']==1 and row['day']==1) or (row['month']==7 and row['day']==4) or (row['month']==11 and row['day']==11) or (row['month']==12 and row['day']==25) or (row['month']==1 and row['day'] >= 15 and row['day'] <= 21 and row['weekday'] == 0) or (row['month']==2 and row['day'] >= 15 and row['day'] <= 21 and row['weekday'] == 0) or (row['month']==5 and row['day'] >= 25 and row['day'] <= 31 and row['weekday'] == 0) or (row['month']==9 and row['day'] >= 1 and row['day'] <= 7 and row['weekday'] == 0) or (row['month']==10 and row['day'] >= 8 and row['day'] <= 14 and row['weekday'] == 0) or (row['month']==11 and row['day'] >= 22 and row['day'] <= 28 and row['weekday'] == 3) else 0, axis=1)
    df['is_day_before_holyday'] = df.apply(lambda row: 1 if (row['month']==12 and row['day']==31) or (row['month']==7 and row['day']==3) or (row['month']==11 and row['day']==10) or (row['month']==12 and row['day']==24) or (row['month']==1 and row['day'] >= 14 and row['day'] <= 20 and row['weekday'] == 6) or (row['month']==2 and row['day'] >= 14 and row['day'] <= 20 and row['weekday'] == 6) or (row['month']==5 and row['day'] >= 24 and row['day'] <= 30 and row['weekday'] == 6) or ((row['month']==9 and row['day'] >= 1 and row['day'] <= 6) or (row['month']==8 and row['day'] == 31) and row['weekday'] == 6) or (row['month']==10 and row['day'] >= 7 and row['day'] <= 13 and row['weekday'] == 6) or (row['month']==11 and row['day'] >= 21 and row['day'] <= 27 and row['weekday'] == 2) else 0, axis=1)
    df.drop('day', axis=1, inplace=True)



# Extract features
print('Extracting train features')
extract_features(train)
print('Extracting test features')
extract_features(test)

x_train = np.array(train.drop(['id', 'pickup_datetime', 'dropoff_datetime', 'store_and_fwd_flag', 'trip_duration'], axis=1))
y_train = np.log(train[TARGET]+1)
mean_trip_duration = np.mean(y_train)

print('X.shape = ' + str(x_train.shape))
print('y.shape = ' + str(y_train.shape))

x_test = np.array(test.drop(['id', 'pickup_datetime', 'store_and_fwd_flag'], axis=1))


kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


et_params = {
    'n_jobs': 16,
    'n_estimators': 10,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 10,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 50
}



rd_params={
    'alpha': 10
}


ls_params={
    'alpha': 0.005
}


xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)

xg_oof_train, xg_oof_test = get_oof(xg)
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)
rd_oof_train, rd_oof_test = get_oof(rd)
ls_oof_train, ls_oof_test = get_oof(ls)

print("XG-CV: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train))))
print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))


x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.1,
    'objective': 'reg:linear',
    'max_depth': 1,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=5, nfold=4, seed=SEED, stratified=False, verbose_eval=1, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = gbdt.predict(dtest)
saleprice = np.exp(submission[TARGET])-1
submission[TARGET] = saleprice
submission.to_csv('xgstacker_starter.sub.csv', index=None)





