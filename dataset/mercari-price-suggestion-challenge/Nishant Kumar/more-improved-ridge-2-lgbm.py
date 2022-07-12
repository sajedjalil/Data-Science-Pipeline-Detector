# Based on Bojan's -> https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944
# Changes:
# 1. Split category_name into sub-categories
# 2. Parallelize LGBM to 4 cores
# 3. Increase the number of rounds in 1st LGBM
# 4. Another LGBM with different seed for model and training split, slightly different hyper-parametes.
# 5. Weights on ensemble
# 6. SGDRegressor doesn't improve the result, going with only 1 Ridge and 2 LGBM
#remove zero price items
import pyximport; pyximport.install()
import gc
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
#svd = TruncatedSVD(n_components=1000, random_state=42)

from joblib import Parallel, delayed

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
import lightgbm as lgb

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000


def rmsle(y, y0):
     assert len(y) == len(y0)
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
    
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    
def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


def main():
    start_time = time.time()

    train = pd.read_table('../input/train.tsv', engine='c')
    test = pd.read_table('../input/test.tsv', engine='c')
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    nrow_test = train.shape[0] #-dftt.shape[0]
    dftt = train[(train.price < 1.0)]
    train = train.drop(train[(train.price < 1.0)].index)
    del dftt['price']
    nrow_train = train.shape[0] #-dftt.shape[0]
    #nrow_test = train.shape[0] + dftt.shape[0]
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, dftt, test])
    submission: pd.DataFrame = test[['test_id']]

    del train
    del test
    gc.collect()
    
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
    zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    merge.drop('category_name', axis=1, inplace=True)
    print('[{}] Split categories completed.'.format(time.time() - start_time))

    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Cut completed.'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))

    cv = TfidfVectorizer(min_df=NAME_MIN_DF,ngram_range=(1, 2),
                         stop_words='english')
    X_name = cv.fit_transform(merge['name'])
    print('[{}] Count vectorize `name` completed.'.format(time.time() - start_time))

    cv = CountVectorizer()
    X_category1 = cv.fit_transform(merge['general_cat'])
    X_category2 = cv.fit_transform(merge['subcat_1'])
    X_category3 = cv.fit_transform(merge['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
    X_description = tv.fit_transform(merge['item_description'])
    print('[{}] TFIDF vectorize `item_description` completed.'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
    print (X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape, X_name.shape)
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()
    print('[{}] Create sparse merge completed'.format(time.time() - start_time))

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    
    model = Ridge(alpha=.5, copy_X=True, fit_intercept=True, max_iter=100,
      normalize=False, random_state=101, solver='auto', tol=0.01)
    model.fit(X, y)
    print('[{}] Train ridge completed'.format(time.time() - start_time))
    predsR = model.predict(X=X_test)
    print('[{}] Predict ridge completed'.format(time.time() - start_time))

    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.15, random_state = 144) 
    d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
    d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
    watchlist = [d_train, d_valid]
    
    params = {
        'learning_rate': 0.65,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': 4
    }

    params2 = {
        'learning_rate': 0.85,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 140,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 2,
        'bagging_fraction': 1,
        'nthread': 4
    }

    model = lgb.train(params, train_set=d_train, num_boost_round=7500, valid_sets=watchlist, \
    early_stopping_rounds=1000, verbose_eval=1000) 
    predsL = model.predict(X_test)
    
    print('[{}] Predict lgb 1 completed.'.format(time.time() - start_time))
    
    train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.1, random_state = 101) 
    d_train2 = lgb.Dataset(train_X2, label=train_y2, max_bin=8192)
    d_valid2 = lgb.Dataset(valid_X2, label=valid_y2, max_bin=8192)
    watchlist2 = [d_train2, d_valid2]

    model = lgb.train(params2, train_set=d_train2, num_boost_round=5000, valid_sets=watchlist2, \
    early_stopping_rounds=500, verbose_eval=500) 
    predsL2 = model.predict(X_test)

    print('[{}] Predict lgb 2 completed.'.format(time.time() - start_time))

    preds = predsR*0.3 + predsL*0.35 + predsL2*0.35

    submission['price'] = np.expm1(preds)
    submission.to_csv("submission_ridge_2xlgbm.csv", index=False)

if __name__ == '__main__':
    main()