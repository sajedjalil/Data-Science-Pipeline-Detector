#===========================================================
# Library
#===========================================================
import os
import gc
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from contextlib import contextmanager
import time

import numpy as np
import pandas as pd
import scipy as sp
import random

from functools import partial

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score

import torch

import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")


#===========================================================
# Utils
#===========================================================
def get_logger(filename='log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = get_logger()


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_everything(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def load_df(path, df_name, debug=False):
    if path.split('.')[-1]=='csv':
        df = pd.read_csv(path)
        if debug:
            df = pd.read_csv(path, nrows=1000)
    elif path.split('.')[-1]=='pkl':
        df = pd.read_pickle(path)
    if logger==None:
        print(f"{df_name} shape / {df.shape} ")
    else:
        logger.info(f"{df_name} shape / {df.shape} ")
    return df


def make_folds(_df, _id, target, fold, group=None, save_path='folds.csv'):
    df = _df.copy()
    if group==None:
        for n, (train_index, val_index) in enumerate(fold.split(df, df[target])):
            df.loc[val_index, 'fold'] = int(n)
    else:
        le = preprocessing.LabelEncoder()
        groups = le.fit_transform(df[group].values)
        for n, (train_index, val_index) in enumerate(fold.split(df, df[target], groups)):
            df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    df[[_id, target, 'fold']].to_csv(save_path, index=None)
    return df[[_id, target, 'fold']]


def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')


class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3
        return X_p

    def coefficients(self):
        return self.coef_['x']


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        logger.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


#===========================================================
# Config
#===========================================================
PARENT_DICT = '../input/data-science-bowl-2019/'
df_path_dict = {'train': PARENT_DICT+'train.csv',
                'test': PARENT_DICT+'test.csv',
                'train_labels': PARENT_DICT+'train_labels.csv', 
                'specs': PARENT_DICT+'specs.csv', 
                'sample_submission': PARENT_DICT+'sample_submission.csv'}
OUTPUT_DICT = ''

ID = 'installation_id'
TARGET = 'accuracy_group'
SEED = 42
seed_everything(seed=SEED)

N_FOLD = 5
Fold = GroupKFold(n_splits=N_FOLD)


#===========================================================
# Feature Engineering
# credits: 
# https://www.kaggle.com/ragnar123/simple-exploratory-data-analysis-and-model
# https://www.kaggle.com/gpreda/data-science-bowl-fast-compact-solution
#===========================================================
def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek  
    return df
    
def get_object_columns(df, columns):
    df = df.groupby(['installation_id', columns])['event_id'].count().reset_index()
    df = df.pivot_table(index = 'installation_id', columns = [columns], values = 'event_id')
    df.columns = list(df.columns)
    df.fillna(0, inplace = True)
    return df

def get_numeric_columns(df, column):
    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std']})
    df.fillna(0, inplace = True)
    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_min', f'{column}_max', f'{column}_std']
    return df

def get_numeric_columns_add(df, agg_column, column):
    df = df.groupby(['installation_id', agg_column]).agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std']}).reset_index()
    df = df.pivot_table(index = 'installation_id', columns = [agg_column], values = [col for col in df.columns if col not in ['installation_id', 'type']])
    df.fillna(0, inplace = True)
    df.columns = list(df.columns)
    return df

def perform_features_engineering(train_df, test_df, train_labels_df):
    print(f'Perform features engineering')
    numerical_columns = ['game_time']
    categorical_columns = ['type', 'world']

    comp_train_df = pd.DataFrame({'installation_id': train_df['installation_id'].unique()})
    comp_train_df.set_index('installation_id', inplace = True)
    comp_test_df = pd.DataFrame({'installation_id': test_df['installation_id'].unique()})
    comp_test_df.set_index('installation_id', inplace = True)

    test_df = extract_time_features(test_df)
    train_df = extract_time_features(train_df)

    for i in numerical_columns:
        comp_train_df = comp_train_df.merge(get_numeric_columns(train_df, i), left_index = True, right_index = True)
        comp_test_df = comp_test_df.merge(get_numeric_columns(test_df, i), left_index = True, right_index = True)
    
    for i in categorical_columns:
        comp_train_df = comp_train_df.merge(get_object_columns(train_df, i), left_index = True, right_index = True)
        comp_test_df = comp_test_df.merge(get_object_columns(test_df, i), left_index = True, right_index = True)
    
    for i in categorical_columns:
        for j in numerical_columns:
            comp_train_df = comp_train_df.merge(get_numeric_columns_add(train_df, i, j), left_index = True, right_index = True)
            comp_test_df = comp_test_df.merge(get_numeric_columns_add(test_df, i, j), left_index = True, right_index = True)
    
    
    comp_train_df.reset_index(inplace = True)
    comp_test_df.reset_index(inplace = True)
    
    print('Our training set have {} rows and {} columns'.format(comp_train_df.shape[0], comp_train_df.shape[1]))

    # get the mode of the title
    labels_map = dict(train_labels_df.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))
    # merge target
    labels = train_labels_df[['installation_id', 'title', 'accuracy_group']]
    # replace title with the mode
    labels['title'] = labels['title'].map(labels_map)
    # get title from the test set
    comp_test_df['title'] = test_df.groupby('installation_id').last()['title'].map(labels_map).reset_index(drop = True)
    # join train with labels
    comp_train_df = labels.merge(comp_train_df, on = 'installation_id', how = 'left')
    print('We have {} training rows'.format(comp_train_df.shape[0]))
    
    return comp_train_df, comp_test_df


#===========================================================
# model
#===========================================================
def run_single_lightgbm(param, train_df, test_df, folds, features, target, fold_num=0, categorical=[]):
    
    trn_idx = folds[folds.fold != fold_num].index
    val_idx = folds[folds.fold == fold_num].index
    logger.info(f'len(trn_idx) : {len(trn_idx)}')
    logger.info(f'len(val_idx) : {len(val_idx)}')
    
    if categorical == []:
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],
                               label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train_df.iloc[val_idx][features],
                               label=target.iloc[val_idx])
    else:
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],
                               label=target.iloc[trn_idx],
                               categorical_feature=categorical)
        val_data = lgb.Dataset(train_df.iloc[val_idx][features],
                               label=target.iloc[val_idx],
                               categorical_feature=categorical)

    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))

    num_round = 10000

    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=1000,
                    early_stopping_rounds=100)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_num

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration)
    
    # RMSE
    logger.info("fold{} RMSE score: {:<8.5f}".format(fold_num, np.sqrt(mean_squared_error(target[val_idx], oof[val_idx]))))
    
    # QWK
    optR = OptimizedRounder()
    optR.fit(oof[val_idx], target[val_idx])
    coefficients = optR.coefficients()
    #coefficients = [0.5, 1.5, 2.5]
    logger.info(f"coefficients: {coefficients}")
    qwk_oof = optR.predict(oof[val_idx], coefficients)
    logger.info("fold{} QWK score: {:<8.5f}".format(fold_num, quadratic_weighted_kappa(qwk_oof, target[val_idx])))
    
    return oof, predictions, fold_importance_df


def run_kfold_lightgbm(param, train, test, folds, features, target, n_fold=5, categorical=[]):
    
    logger.info(f"================================= {n_fold}fold lightgbm =================================")
    
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    for fold_ in range(n_fold):
        print("Fold {}".format(fold_))
        _oof, _predictions, fold_importance_df = run_single_lightgbm(param,
                                                                     train,
                                                                     test,
                                                                     folds,
                                                                     features,
                                                                     target,
                                                                     fold_num=fold_,
                                                                     categorical=categorical)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        oof += _oof
        predictions += _predictions / n_fold

    # RMSE
    logger.info("CV RMSE score: {:<8.5f}".format(np.sqrt(mean_squared_error(target, oof))))
    
    # QWK
    optR = OptimizedRounder()
    optR.fit(oof, target)
    coefficients = optR.coefficients()
    #coefficients = [0.5, 1.5, 2.5]
    logger.info(f"coefficients: {coefficients}")
    qwk_oof = optR.predict(oof, coefficients)
    logger.info("CV QWK score: {:<8.5f}"
                .format(quadratic_weighted_kappa(qwk_oof, target)))
    qwk_predictions = optR.predict(predictions, coefficients)
    
    submission = pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": qwk_predictions})
    submission[TARGET] = submission[TARGET].astype(int)
    submission.to_csv(OUTPUT_DICT+'submission.csv', index=False)
    feature_importance_df.to_csv(OUTPUT_DICT+'feature_importance_df_lightgbm.csv', index=False)

    logger.info(f"=========================================================================================")


#===========================================================
# main
#===========================================================
def main():
    
    DEBUG = False
    
    with timer('Data Loading'):
        train = load_df(path=df_path_dict['train'], df_name='train', debug=DEBUG)
        train = reduce_mem_usage(train)
        train_labels = load_df(path=df_path_dict['train_labels'], df_name='train_labels', debug=DEBUG)
        test = load_df(path=df_path_dict['test'], df_name='test', debug=DEBUG)
        test = reduce_mem_usage(test)
        #specs = load_df(path=df_path_dict['specs'], df_name='specs')
        sample_submission = load_df(path=df_path_dict['sample_submission'], df_name='sample_submission')
    
    with timer('Creating features'):
        train_df, test_df = perform_features_engineering(train, test, train_labels)
        del train, test, train_labels; gc.collect()
        train_df = reduce_mem_usage(train_df)
        test_df = reduce_mem_usage(test_df)
        logger.info(f'train_df shape : {train_df.shape}')
        train_df.to_csv('train.csv', index=False)
        logger.info(f'test_df shape : {test_df.shape}')
        test_df.to_csv('test.csv', index=False)
        
    with timer('Run lightgbm'):
        lgb_param = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'data_random_seed': SEED,
                'max_depth': -1,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'min_data_in_leaf': 100,
            }
        logger.info(f"lgb_param : {lgb_param}")
        
        target = train_df[TARGET]
        folds = make_folds(train_df, ID, TARGET, Fold, group='installation_id')
        test_df = pd.concat([sample_submission.set_index('installation_id').drop(columns=['accuracy_group']), 
                             test_df.set_index('installation_id')], axis=1).reset_index()
        
        num_features = [c for c in test_df.columns if test_df.dtypes[c] != 'object']
        cat_features = ['title']
        features = num_features + cat_features
        drop_features = [ID, TARGET, 'accuracy']
        features = [c for c in features if c not in drop_features]
        logger.info(features)
        
        if cat_features:
            for c in cat_features:
                le = LabelEncoder()
                le.fit(train_df[c])
                train_df[c] = le.transform(train_df[c])
                test_df[c] = le.transform(test_df[c])
        
        run_kfold_lightgbm(lgb_param, train_df, test_df, folds, features, target, n_fold=N_FOLD, categorical=cat_features)


if __name__ == "__main__":
    main()