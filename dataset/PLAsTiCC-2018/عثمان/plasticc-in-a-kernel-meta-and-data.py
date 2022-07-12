"""
PLAsTiCC_in_a_kernel_meta_and_data
----------------------------------
@website https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data

@author Olivier https://www.kaggle.com/ogrellier

Goal :
------
Train 5 lightgbms on the meta_data + aggregated data

Then go through test data in chunks and generate predictions

New in this version :
---------------------
1. This versions adds some of the Flux calculations made available by MichaelApers https://www.kaggle.com/michaelapers
    here https://www.kaggle.com/michaelapers/the-plasticc-astronomy-starter-kit
2. class 99 mean adjustment

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import logging

galatic = [92, 65, 16,  6, 53]

def create_logger():
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler('simple_lightgbm.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger():
    return logging.getLogger('main')


def lgb_multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    
    classes = [x for x in classes if x not in galatic]
    for cls in galatic: del class_weight[cls]
    
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False


def multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    
    classes = [x for x in classes if x not in galatic]
    for cls in galatic: del class_weight[cls]
    
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


def predict_chunk(df_, clfs_, meta_, features, train_mean):

    df_['flux_ratio_sq'] = np.power(df_['flux'] / df_['flux_err'], 2.0)
    df_['flux_by_flux_ratio_sq'] = df_['flux'] * df_['flux_ratio_sq']

    # Group by object id
    aggs = get_aggregations()

    aggs = get_aggregations()
    aggs['flux_ratio_sq'] = ['sum']
    aggs['flux_by_flux_ratio_sq'] = ['sum']

    new_columns = get_new_columns(aggs)

    agg_ = df_.groupby('object_id').agg(aggs)
    agg_.columns = new_columns

    agg_ = add_features_to_agg(df=agg_)

    # Merge with meta data
    full_test = agg_.reset_index().merge(
        right=meta_,
        how='left',
        on='object_id'
    )

    full_test = full_test.fillna(train_mean)
    # Make predictions
    preds_ = None
    for clf in clfs_:
        if preds_ is None:
            preds_ = clf.predict_proba(full_test[features]) / len(clfs_)
        else:
            preds_ += clf.predict_proba(full_test[features]) / len(clfs_)

    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds_.shape[0])
    for i in range(preds_.shape[1]):
        preds_99 *= (1 - preds_[:, i])

    # Create DataFrame from predictions
    preds_df_ = pd.DataFrame(preds_, columns=['class_' + str(s) for s in clfs_[0].classes_])
    preds_df_['object_id'] = full_test['object_id']
    preds_df_['class_99'] = 0.14 * preds_99 / np.mean(preds_99) 

    print(preds_df_['class_99'].mean())

    del agg_, full_test, preds_
    gc.collect()

    return preds_df_


def save_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    plt.figure(figsize=(8, 12))
    sns.barplot(x='gain', y='feature', data=importances_.sort_values('mean_gain', ascending=False))
    plt.tight_layout()
    plt.savefig('importances.png')


def train_classifiers(full_train=None, y=None):

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clfs = []
    importances = pd.DataFrame()
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 14,
        'metric': 'multi_logloss',
        'learning_rate': 0.03,
        'subsample': .9,
        'colsample_bytree': .7,
        'reg_alpha': .01,
        'reg_lambda': .01,
        'min_split_gain': 0.01,
        'min_child_weight': 10,
        'n_estimators': 1000,
        'silent': -1,
        'verbose': -1,
        'max_depth': 3
    }
    
    # Compute weights
    w = y.value_counts()
    weights = {i : np.sum(w) / w[i] for i in w.index}
        
    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]

        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgb_multi_weighted_logloss,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )
        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        get_logger().info(multi_weighted_logloss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_)))

        imp_df = pd.DataFrame()
        imp_df['feature'] = full_train.columns
        imp_df['gain'] = clf.feature_importances_
        imp_df['fold'] = fold_ + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

        clfs.append(clf)

    get_logger().info('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_true=y, y_preds=oof_preds))

    return clfs, importances


def get_aggregations():
    return {
        # Dropped mjd aggregations on CPMP advice
        # see https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696
        # 'mjd': ['min', 'max', 'size'],
        'passband': ['mean', 'std', 'var'],  # ''min', 'max', 'mean', 'median', 'std'],
        'flux': ['min', 'max', 'mean', 'median', 'std'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std'],
        'detected': ['mean'],  # ''min', 'max', 'mean', 'median', 'std'],
    }


def get_new_columns(aggs):
    return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


def add_features_to_agg(df):
    # CPMP using the following feature was really silliy :)
    # df['mjd_diff'] = df['mjd_max'] - df['mjd_min']
    # see https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696
    
    # The others may be useful
    df['flux_diff'] = df['flux_max'] - df['flux_min']
    df['flux_dif2'] = (df['flux_max'] - df['flux_min']) / df['flux_mean']
    df['flux_w_mean'] = df['flux_by_flux_ratio_sq_sum'] / df['flux_ratio_sq_sum']
    df['flux_dif3'] = (df['flux_max'] - df['flux_min']) / df['flux_w_mean']

    # del df['mjd_max'], df['mjd_min']

    return df

def main():
    train = pd.read_csv('../input/training_set.csv')
    
    train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
    train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']

    # train = pd.concat([train, pd.get_dummies(train['passband'], prefix='passband')], axis=1, sort=False)

    aggs = get_aggregations()
    aggs['flux_ratio_sq'] = ['sum']
    aggs['flux_by_flux_ratio_sq'] = ['sum']

    # passbands = [f for f in train if 'passband_' in f]
    # get_logger().info('Passband features : {}'.format(passbands))
    # for pb in passbands:
    #     aggs[pb] = ['mean']

    agg_train = train.groupby('object_id').agg(aggs)
    new_columns = get_new_columns(aggs)
    agg_train.columns = new_columns

    agg_train = add_features_to_agg(df=agg_train)
    
    agg_train.head()

    del train
    gc.collect()

    meta_train = pd.read_csv('../input/training_set_metadata.csv')
    meta_train = meta_train[~meta_train.target.isin(galatic)]
    meta_train.head()

    full_train = agg_train.reset_index().merge(
        right=meta_train,
        how='inner',
        on='object_id'
    )

    y = full_train['target']
    del full_train['target']
    del full_train['object_id'], full_train['hostgal_specz']  # , full_train['distmod']

    train_mean = full_train.mean(axis=0)
    full_train.fillna(train_mean, inplace=True)
    get_logger().info(full_train.columns)
    clfs, importances = train_classifiers(full_train, y)

    save_importances(importances_=importances)

if __name__ == '__main__':
    gc.enable()
    create_logger()
    try:
        main()
    except Exception:
        get_logger().exception('Unexpected Exception Occured')
        raise
