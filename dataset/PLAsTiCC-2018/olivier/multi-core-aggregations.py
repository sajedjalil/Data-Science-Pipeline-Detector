"""
multi_core_aggregations_test
----------------------------------
@website https://www.kaggle.com/ogrellier/multi_core_aggregations

@author Olivier https://www.kaggle.com/ogrellier

Goal :
------
This is an example of aggregation on multiple cores
The idea is to create groups and then dispatch them on different core to parallelize the process

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import logging
import time
import multiprocessing as mp
from functools import partial


def my_parallel_apply(groups=None, func=None, num_workers=4):
    """Creates a pool of workers and dispatch the list of objects to a function that will aggregate them"""
    with mp.pool.Pool(num_workers) as executor:
        the_aggs = executor.map(func, groups)
    
    return pd.concat([a for a in the_aggs if not a.empty], axis=0)


def multi_agg(group=None, the_aggs=None):
    return group.agg(the_aggs)


def parallel_apply(group=None, func=None):
    return group.apply(func)


def compute_all_aggregated_features(df):
    # Compute weighted mean
    a_s = df['flux'] * np.power(df['flux'] / df['flux_err'], 2)
    b_s = np.power(df['flux'] / df['flux_err'], 2)
    wmean = np.sum(a_s) / np.sum(b_s)

    flux_med = np.median(df['flux'])
    # Compute normed flux
    normed_flux = (df['flux'].max() - df['flux'].min()) / wmean

    # normed_median_flux
    normed_median_flux = np.median(np.abs(df['flux'] - flux_med) / wmean)
    
    return pd.Series([wmean, flux_med, normed_flux, normed_median_flux])
    

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


class PlasticcPredictor(object):

    def __init__(self, trn_file, trn_meta_file, sub_file, sub_meta_file, lgb_params):

        self.trn_file = trn_file
        self.trn_meta_file = trn_meta_file
        self.sub_file = sub_file
        self.sub_meta_file = sub_meta_file
        self.aggs = self._get_initial_aggregations()
        self.lgb_params = lgb_params
        self.clfs = None
        self.importances = None
        self.fill_nan = None
        self.features = None
        self.indexers = None

    def train(self):
        # Get train data
        np.random.seed(101)
        full_train = self._get_dataset(
            df=pd.read_csv(self.trn_file),
            meta_df=pd.read_csv(self.trn_meta_file)
        )

        # Get target
        y = full_train['target']
        del full_train['target']

        # Remove features not used for training
        del full_train['object_id'], full_train['hostgal_specz']

        # Fill NaNs
        self.fill_nan = full_train.mean(axis=0)
        full_train.fillna(self.fill_nan, inplace=True)
        get_logger().info(full_train.columns)
        self.features = full_train.columns

        # Train classifiers
        self._fit(full_train, y)

        # Store Feature importances
        self._save_importances()

    def _fit(self, df, y):

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        self.clfs = []
        self.importances = pd.DataFrame()
        
        w = y.value_counts()
        weights = {i : np.sum(w) / w[i] for i in w.index}
        
        oof_preds = np.zeros((len(df), np.unique(y).shape[0]))
        for fold_, (val_, trn_) in enumerate(folds.split(y, y)):
            trn_x, trn_y = df.iloc[trn_], y.iloc[trn_]
            val_x, val_y = df.iloc[val_], y.iloc[val_]

            clf = lgb.LGBMClassifier(**self.lgb_params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric=lgb_multi_weighted_logloss,
                sample_weight=trn_y.map(weights),
                verbose=100,
                early_stopping_rounds=200
            )
            oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
            get_logger().info(
                multi_weighted_logloss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_)))

            imp_df = pd.DataFrame()
            imp_df['feature'] = df.columns
            imp_df['gain'] = clf.feature_importances_
            imp_df['fold'] = fold_ + 1
            self.importances = pd.concat([self.importances, imp_df], axis=0, sort=False)

            self.clfs.append(clf)

        get_logger().info('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_true=y, y_preds=oof_preds))

    def predict(self):
        meta_test = pd.read_csv(self.sub_meta_file)

        start = time.time()
        chunks = 10000000
        remain_df = None

        for i_c, df in enumerate(pd.read_csv(self.sub_file, chunksize=chunks, iterator=True)):
            # Check object_ids
            # I believe np.unique keeps the order of group_ids as they appear in the file
            unique_ids = np.unique(df['object_id'])
            new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()

            if remain_df is None:
                df = df.loc[df['object_id'].isin(unique_ids[:-1])].copy()
            else:
                df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)

            # Create remaining samples df
            remain_df = new_remain_df

            # Get predictions
            preds_df = self._predict_chunk(df=df, meta_df=meta_test)

            if i_c == 0:
                preds_df.to_csv('predictions_refactored.csv', header=True, index=False, float_format='%.6f')
            else:
                preds_df.to_csv('predictions_refactored.csv', header=False, mode='a', index=False, float_format='%.6f')

            del preds_df
            gc.collect()

            get_logger().info('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

            get_logger().info('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
            print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
            
            # if i_c == 2:
            #     break

        # Compute last object in remain_df
        preds_df = self._predict_chunk(df=remain_df, meta_df=meta_test)

        preds_df.to_csv('predictions_refactored.csv', header=False, mode='a', index=False, float_format='%.6f')

        z = pd.read_csv('predictions_refactored.csv')

        z = z.groupby('object_id').mean()

        z.to_csv('single_predictions_refactored.csv', index=True, float_format='%.6f')

    def _predict_chunk(self, df, meta_df):

        full_test = self._get_dataset(df=df, meta_df=meta_df)
        full_test = full_test.fillna(self.fill_nan)

        # Make predictions
        preds_ = None
        for clf in self.clfs:
            if preds_ is None:
                preds_ = clf.predict_proba(full_test[self.features]) / len(self.clfs)
            else:
                preds_ += clf.predict_proba(full_test[self.features]) / len(self.clfs)

        # Compute class 99
        preds_99 = np.ones(preds_.shape[0])
        for i in range(preds_.shape[1]):
            preds_99 *= (1 - preds_[:, i])

        # Create DataFrame from predictions
        preds_df_ = pd.DataFrame(preds_, columns=['class_' + str(s) for s in self.clfs[0].classes_])
        preds_df_['object_id'] = full_test['object_id']
        galactic = (full_test['hostgal_photoz'] == 0)
        preds_df_.loc[galactic, 'class_99'] = 0.0148 * preds_99[galactic.values] / np.mean(preds_99)
        preds_df_.loc[~galactic, 'class_99'] = 0.149 * preds_99[~galactic.values] / np.mean(preds_99)

        del full_test, preds_
        gc.collect()

        return preds_df_

    def _get_dataset(self, df, meta_df):

        # May be necessary depending on your features 
        df = df.sort_values(['object_id', 'mjd'])

        # First create the list of object_ids
        # For some reason you cannot split GroupBy object directly, which is a pain
        # So I had to create groups myself
        
        # 1. create an array containing all object ids
        object_ids = np.unique(df['object_id'])
        
        # 2. Split these objects in the number of cores you target
        id_groups = np.array_split(ary=object_ids, indices_or_sections=4)

        # 3. Create a list of GroupBy objects corresponding to the above list of objects 
        groups = [
            df.loc[df['object_id'].isin(obj)].groupby('object_id')
            for obj in id_groups
        ]

        # 4. Go to multi-processing
        agg_df = my_parallel_apply(
            # Give the list of 4 GroupBy Objects
            groups=groups,
            # The functools.partial is used to pass the aggregations to the multi_agg function
            func=partial(multi_agg, the_aggs=self.aggs),
            # Provide the number of cores you target
            num_workers=4
        )
        
        # Change the columns names
        agg_df.columns = self._get_new_columns(self.aggs)
        
        # So that was for very simple aggregations
        # Let's go to something a bit more tricky like using multiple columns at once
        
        z = my_parallel_apply(
            groups=groups,
            func=partial(parallel_apply, func=compute_all_aggregated_features),
            num_workers=4
        )
        # Add columns one by one, I let you replace this by a simple pd.concat
        for f in z.columns:
            agg_df['other_' + str(f)] = z[f]
        

        get_logger().info('aggregation done')

        del groups
        gc.collect()

        # Free up memory
        del df
        gc.collect()

        # Merge datasets
        full_df = agg_df.reset_index().merge(
            right=meta_df,
            how='left',
            on='object_id'
        )

        # Free memory
        del meta_df, agg_df
        gc.collect()

        return full_df

    def _save_importances(self):
        mean_gain = self.importances[['gain', 'feature']].groupby('feature').mean()
        self.importances['mean_gain'] = self.importances['feature'].map(mean_gain['gain'])
        plt.figure(figsize=(8, 20))
        sns.barplot(x='gain', y='feature', data=self.importances.sort_values('mean_gain', ascending=False))
        plt.tight_layout()
        plt.savefig('importances_refactored.png')

    @staticmethod
    def _get_initial_aggregations():
        return {
            # 'passband': ['size', 'mean', 'std', 'var', 'skew'],
            'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
            'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
            'detected': ['mean'],
        }

    @staticmethod
    def _get_new_columns(a_dict):
        return [k + '_' + agg for k in a_dict.keys() for agg in a_dict[k]]


if __name__ == '__main__':
    gc.enable()
    create_logger()
    try:
        # Create solution object
        plastiic = PlasticcPredictor(
            trn_file='../input/training_set.csv',
            trn_meta_file='../input/training_set_metadata.csv',
            sub_file='../input/test_set.csv',
            sub_meta_file='../input/test_set_metadata.csv',
            lgb_params={
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 14,
                'metric': 'multi_logloss',
                'learning_rate': .1,
                'subsample': .9,
                'colsample_bytree': .3,
                'reg_alpha': .1,
                'reg_lambda': .01,
                'min_split_gain': 0.05,
                'min_child_weight': 10,
                'n_estimators': 500, 
                'silent': -1,
                'verbose': -1,
                'max_depth': -1,
                'num_leaves': 60,
            }

        )
        # Train the solution
        plastiic.train()
        # Predict
        plastiic.predict()

    except Exception:
        get_logger().exception('Unexpected Exception Occurred')
        raise
