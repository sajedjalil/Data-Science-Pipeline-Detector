# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb

NUM_BRANDS = 4100
NUM_CATEGORIES = 1100
MAX_FEATURES_ITEM_DESCRIPTION = 41000

def handle_missing_data(data):
    # data['category_name'].fillna(value="missing_val",inplace=True)
    data['cat1'].fillna(value="missing_val",inplace=True)
    data['cat2'].fillna(value="missing_val",inplace=True)
    data['cat3'].fillna(value="missing_val",inplace=True)
    data['brand_name'].fillna(value="missing_val",inplace=True)
    data['item_description'].replace('No description yet','missing_val', inplace=True)
    data['item_description'].fillna(value="missing_val",inplace=True)
    
def to_categorical(data):
    # data['category_name'] = data['category_name'].astype('category')
    data['cat1'] = data['cat1'].astype('category')
    data['cat2'] = data['cat2'].astype('category')
    data['cat3'] = data['cat3'].astype('category')
    data['brand_name'] = data['brand_name'].astype('category')
    data['item_condition_id'] = data['item_condition_id'].astype('category')
    
#Accessing with .loc[index,column_name] as it is label based    
def cut(data):
    brand = data['brand_name'].value_counts().loc[lambda x: x.index != 'missing_val'].index[:NUM_BRANDS]
    data.loc[~data['brand_name'].isin(brand), 'brand_name'] = 'missing_val'
    cat1 = data['cat1'].value_counts().loc[lambda x: x.index != 'missing_val'].index[:NUM_CATEGORIES]
    cat2 = data['cat2'].value_counts().loc[lambda x: x.index != 'missing_val'].index[:NUM_CATEGORIES]
    cat3 = data['cat3'].value_counts().loc[lambda x: x.index != 'missing_val'].index[:NUM_CATEGORIES]
    # category = data['category_name'].value_counts().loc[lambda x: x.index != 'missing_val'].index[:NUM_CATEGORIES]
    # data.loc[~data['category_name'].isin(category), 'category_name'] = 'missing_val'
    data.loc[~data['cat1'].isin(cat1), 'cat1'] = 'missing_val'
    data.loc[~data['cat2'].isin(cat2), 'cat2'] = 'missing_val'
    data.loc[~data['cat3'].isin(cat3), 'cat3'] = 'missing_val'

def split_category(x):
    try:
        return x.split("/")
    except:
        return("Blank","Blank","Blank")

    
def main():
    start_time = time.time()
    train = pd.read_table('../input/train.tsv', engine='c')
    test = pd.read_table('../input/test.tsv', engine='c')
    
    print('[{}] Finished loading data '.format(time.time()-start_time))
    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    merge = pd.concat([train, test])
    submission = test[['test_id']]
    
    del train
    del test
    gc.collect()
    
    merge['cat1'], merge['cat2'], merge['cat3'] = zip(*merge['category_name'].apply(lambda x: split_category(x)))
    merge.drop("category_name",axis=1,inplace=True)

    handle_missing_data(merge)
    cut(merge)
    to_categorical(merge)
    
    cv = CountVectorizer(min_df=10,ngram_range=(1, 2),stop_words='english')
    X_name = cv.fit_transform(merge['name'])
    
    cv = CountVectorizer()
    # X_category = cv.fit_transform(merge['category_name'])
    X_cat1 = cv.fit_transform(merge['cat1'])
    X_cat2 = cv.fit_transform(merge['cat2'])
    X_cat3 = cv.fit_transform(merge['cat3'])
    
    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,ngram_range=(1, 3),stop_words='english')
    X_description = tv.fit_transform(merge['item_description'])
    
    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    
    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],sparse=True).values)
    
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_cat1, X_cat2, X_cat3, X_name)).tocsr()
    # print('Finished creating sparse merge matrix')
    
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    
    d_train = lgb.Dataset(X, label=y)
    
    
    params1 = {
        'learning_rate': 0.5,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 64,
        'verbosity': -1,
        'metric': 'RMSE',
        'bagging_fraction': 0.5,
        'nthread': 4
    }
    
    
    params2 = {
        'learning_rate': 0.8,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 128,
        'verbosity': -1,
        'metric': 'RMSE',
        'bagging_fraction': 1,
        'nthread': 4
    }
    
    model = lgb.train(params1, train_set=d_train, num_boost_round=6000, verbose_eval=100)
    preds = 0.25*model.predict(X_test)
    
    
    model = lgb.train(params2, train_set=d_train, num_boost_round=3000, verbose_eval=100)
    preds += 0.35*model.predict(X_test)
    
    model = Ridge(solver="sag", random_state=205)
    model.fit(X, y)
    preds += 0.4*model.predict(X=X_test)
    
    print('[{}] Prediction completed.'.format(time.time() - start_time))

    submission['price'] = np.expm1(preds)
    submission.to_csv("submission_2lgbm_with_ridge_updated.csv", index=False)
    
if __name__ == '__main__':
    main()