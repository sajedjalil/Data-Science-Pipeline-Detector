#!/usr/bin/env python3

"""
hyperparameters in CB_TRAIN_PARAMS as presented in this script are tuned to make it possible to run this script 
in resource constrained Kaggle containers (16G RAM).

If RAM (approx. 100G) and CPU time (approx. 3.5h on modern Xeon) is available public LB score of 0.230 is achievable 
using the following:

CB_TRAIN_PARAMS = {
    'eval_metric' : 'RMSE',
    'depth' : 10,
    'iterations' : 6400
}

"""

import gc
import sys

# python2 compatibility
if sys.version_info[0] == 3:
    unicode = str


import catboost as cb
import pandas as pd
import numpy as np
import sklearn.model_selection


# configuration

DEBUG = False # test on small sample for debugging


PROCESSED_DATA_NAME = 'simple1' # name for saving preprocesed data
 
OUTPUT_NAME = 'simple1_hp_tuned' # name for final results - model and predictions on test set


INPUT_PATH = '../input/'


SAVE_PREPROCESSED_DATA = False # False because of Kaggle's limited disk space, prefer True when running locally
USE_PREPROCESSED_DATA = False # use stored preprocessed data, don't rerun preprocessing

SAVE_MODEL = False # False because of Kaggle's limited disk space, prefer True when running locally


VALIDATE = False # run with validation instead of using all train set for training

TRAIN_SHARE = 0.9 # share of train set (the rest is held out as the validation set) from all train set


CB_TECH_PARAMS = {
    'verbose' : True,
    'random_seed' : 42,
    'save_snapshot' : False, # False because of Kaggle's limited disk space, prefer True when running locally
    'allow_writing_files' : False, # False because of Kaggle's limited disk space, prefer True when running locally
    'used_ram_limit' : 15*(2 ** 30),
}


CB_TRAIN_PARAMS = {
    'eval_metric' : 'RMSE',
    'learning_rate' : 0.1,
    'depth' : 10,
    'iterations' : 560,
    'leaf_estimation_iterations' : 8
}

CAT_FEATURE_NAMES = [
    'user_id',
    'region',
    'city',
    'parent_category_name',
    'category_name',
    'param_1',
    'param_2',
    'param_3',
    'user_type',
    
    # computed
    'has_image',
    'activation_date_weekday',
]

NON_CAT_FEATURE_NAMES = [
    'price',
    'item_seq_number',
    
    # computed
    'title_length',
    'description_length'
]

TARGET_NAME = 'deal_probability'



DEBUG_SIZE = 50000

################################################################

NAME_PREFIX = 'debug.' if DEBUG else ''

PROCESSED_DATA_PREFIX = NAME_PREFIX + PROCESSED_DATA_NAME + '.'
OUTPUT_PREFIX = NAME_PREFIX + OUTPUT_NAME + '.'


CB_TECH_PARAMS['snapshot_file'] = OUTPUT_PREFIX + 'catboost_snapshot'

CB_PARAMS = CB_TECH_PARAMS
CB_PARAMS.update(CB_TRAIN_PARAMS)


FEATURE_NAMES = CAT_FEATURE_NAMES + NON_CAT_FEATURE_NAMES

################################################################


# prefix is either 'train' or 'test'
# returns X, Y, None         for 'train'
# returns X, None, item_ids  for 'test' 
def preprocess_data(prefix):
    def log(msg):
        print ('preprocess_data. prefix "', prefix, '" ', msg)
    
    log('start')
    
    cols = {
        #'item_id'
        'user_id': str,
        'region': unicode,
        'city': unicode,
        'parent_category_name': unicode,
        'category_name': unicode,
        'param_1': unicode,
        'param_2': unicode,
        'param_3': unicode,
        'title' : unicode,
        'description' : unicode,
        'price' : np.float32,
        'item_seq_number': np.uint32,
        'activation_date': object, # in fact yyyy-mm-dd date
        'user_type': str,
        'image': str,
        'image_top_1': np.float32,
        
    }
    
    if prefix == 'train':
        cols['deal_probability'] = np.float32
    elif prefix == 'test':
        cols['item_id'] = str
    else:
        raise Exception('prefix is neither train nor test')
    
    df = pd.read_csv(
        INPUT_PATH + prefix + '.csv',
        nrows=DEBUG_SIZE if DEBUG else None,
        usecols=cols.keys(),
        dtype=cols,
        encoding = 'utf8'
    )
    
    log('data after loading')
    print (df.info())
    print (df.head(5))
    
    # comp features
    
    # nans in 'image' column are parsed as floats
    df['has_image'] = df.image.apply(lambda image: True if type(image) == unicode else False).astype('bool')
    df.drop(['image'], axis=1, inplace=True)
    gc.collect()
    
    df['activation_date_weekday'] = pd.to_datetime(df.activation_date).apply(lambda ad: ad.weekday()).astype('uint8')
    df.drop(['activation_date'], axis=1, inplace=True)
    gc.collect()

    for col in ['title', 'description']:
        df[col + '_length'] = df[col].apply(lambda txt: len(txt) if type(txt) == unicode else 0).astype('uint32')
        df.drop([col], axis=1, inplace=True)
    gc.collect()
    
    # fix for catboost's python2 version unicode incompatibility
    for col, coltype in cols.items():
        if (coltype == unicode) and (col in df):
            df[col] = df[col].apply(
                (lambda utxt: utxt.encode('ascii', 'xmlcharrefreplace') if type(utxt) == unicode else utxt) if sys.version_info[0] == 2
                else (lambda utxt: utxt.encode('ascii', 'xmlcharrefreplace').decode('ascii') if type(utxt) == unicode else utxt)
            ).astype('str')
    gc.collect()
    
    log('data after comp features')
    print (df.info())
    print (df.head(5))
    
    log('end')

    return (df[FEATURE_NAMES].values,
            (df[TARGET_NAME].values if prefix == 'train' else None),
            (df['item_id'].values if prefix == 'test' else None))


# prefix is either 'train' or 'test'
# returns X, Y, None         for 'train'
# returns X, None, item_ids  for 'test' 
def get_preprocessed_data(prefix):
    pp_prefix = PROCESSED_DATA_PREFIX + prefix + '.'
    if USE_PREPROCESSED_DATA:
        return (np.load(pp_prefix + 'X.npy'),
                (np.load(pp_prefix + 'Y.npy') if prefix == 'train' else None),
                (np.load(pp_prefix + 'item_ids.npy') if prefix == 'test' else None))
    else:
        X, Y, item_ids = preprocess_data(prefix)
        if SAVE_PREPROCESSED_DATA:
            for name in ('X', 'Y') if prefix == 'train' else ('X', 'item_ids'):
                np.save(pp_prefix + name + '.npy', eval(name))
        return X, Y, item_ids


"""
  returns (cb_model, val_pool)
     val_pool can be useful later of additional metrics evaluation

     pass None as val_X and val_Y to train w/o eval_set
"""
def train_model(train_X, train_Y, val_X, val_Y, feature_names, cat_feature_names):
    print ('start training')
    
    print ('feature_names', feature_names)
    print ('cat_feature_names', cat_feature_names)
    
    cat_feature_indices = [feature_names.index(cat_name) for cat_name in cat_feature_names]
    
    train_pool = cb.Pool(
        train_X,
        label=train_Y,
        feature_names=feature_names,
        cat_features=cat_feature_indices
    )
    
    if (val_X is not None) and (val_Y is not None):
        val_pool = cb.Pool(
            val_X,
            label=val_Y,
            feature_names=feature_names,
            cat_features=cat_feature_indices
        )
    else:
        val_pool = None
    
    cb_model = cb.CatBoostRegressor(**CB_PARAMS)
    cb_model.fit(train_pool, eval_set=val_pool)
    if SAVE_MODEL:
        cb_model.save_model(OUTPUT_PREFIX + 'model.cbm')
    
    del train_pool
    gc.collect()
    
    print ('end training')
    
    return (cb_model, val_pool)


def predict(item_ids, test_X, cb_model):
    print ('start predict')
    
    sub = pd.DataFrame()
    sub['item_id'] = item_ids
    
    sub['deal_probability'] = np.clip( cb_model.predict(test_X), 0.0, 1.0)
    
    sub.to_csv(OUTPUT_PREFIX + 'submission.csv', index=False)
    
    print ('end predict')
    
################################################################


alltrain_X, alltrain_Y, _ = get_preprocessed_data('train')

if VALIDATE:
    train_X, val_X, train_Y, val_Y = sklearn.model_selection.train_test_split(
        alltrain_X, alltrain_Y, train_size=TRAIN_SHARE, random_state=99
    )
else:
    train_X, val_X, train_Y, val_Y = alltrain_X, None, alltrain_Y, None


cb_model, _ = train_model(train_X, train_Y, val_X, val_Y, FEATURE_NAMES, CAT_FEATURE_NAMES)

test_X, _, item_ids = get_preprocessed_data('test')

predict(item_ids, test_X, cb_model)




