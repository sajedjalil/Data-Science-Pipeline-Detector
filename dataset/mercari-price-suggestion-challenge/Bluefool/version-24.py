import gc
import time
import numpy as np
import pandas as pd
import re, math
import psutil
import os
import sys
import time
from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

MAX_FEATURES_ITEM_DESCRIPTION = None

start = time.clock()

def cpuStats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)
    print('time in minutes:', (time.clock() - start)/60)


def rmsle(y, preds):
    return np.sqrt(np.square(np.log(preds + 1) - np.log(y + 1)).mean())

def split_cat(cats):
    cat2 = "no category 2"
    cat3 = "no category 3"
    if pd.notnull(cats):
        all_cats = cats.split('/')
        if len(all_cats) > 0:
            cat3 = all_cats[2]
            cat2 = all_cats[0]
    return cat2.lower(), cat3.lower()

def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)

def fill_brands(dataset, top, test):
    brands = pd.concat([dataset['brand_name'], test['brand_name'], top['brand_name']], axis=0).unique().astype(str)
    print(pd.isnull(dataset['brand_name']).sum())
    brands_str = re.compile(r'\b(?:%s)\b' % '|'.join(brands))
    dataset['brand_name'] = dataset.apply(lambda row: row['brand_name'] if pd.notnull(row['brand_name']) or brands_str.match(row['name']) is None else brands_str.match(row['name']).group(0), axis=1)
    test['brand_name'] = test.apply(lambda row: row['brand_name'] if pd.notnull(row['brand_name']) or brands_str.match(row['name']) is None else brands_str.match(row['name']).group(0), axis=1)
    top['brand_name'] = top.apply(lambda row: row['brand_name'] if pd.notnull(row['brand_name']) or brands_str.match(row['name']) is None else brands_str.match(row['name']).group(0), axis=1)
    print(pd.isnull(dataset['brand_name']).sum())
    del brands
    del brands_str
    gc.collect()

def to_lower(dataset):
    
    dataset['category_name'] = dataset['category_name'].str.lower()
    dataset['brand_name'] = dataset['brand_name'].str.lower()
    dataset['brand_name'] = dataset['brand_name'].str.replace("'","")
    dataset['name'] = dataset['name'].str.lower()
    dataset['name'] = dataset['name'].str.replace("'","")
    dataset['item_description'] = dataset['item_description'].str.lower()

def process_dataset(dataset, top, test):
    to_lower(dataset)
    to_lower(test)
    to_lower(top)
    fill_brands(dataset, top, test)
    handle_missing_inplace(dataset)
    handle_missing_inplace(test)
    handle_missing_inplace(top)


def main():
    train = pd.read_table('../input/train.tsv', engine='c')
    train = train[train.price != 0].reset_index(drop=True)
    test = pd.read_table('../input/test.tsv', engine='c')
    print(test.shape)
    test_top = test[:550000].reset_index(drop=True)
    test_rest = test[550000:].reset_index(drop=True)
    print(test_top.shape)
    print(test_rest.shape)
    del test
    gc.collect()
    test_top_predsa: pd.DataFrame = test_top[['test_id']].reset_index(drop=True)
    test_rest_predsa: pd.DataFrame = test_rest[['test_id']].reset_index(drop=True)
    train[['cat2','cat3']] = pd.DataFrame(train.category_name.apply(split_cat).tolist(), columns = ['cat2','cat3'])
    test_top[['cat2','cat3']] = pd.DataFrame(test_top.category_name.apply(split_cat).tolist(), columns = ['cat2','cat3'])
    test_rest[['cat2','cat3']] = pd.DataFrame(test_rest.category_name.apply(split_cat).tolist(), columns = ['cat2','cat3'])
    count_cat3 = pd.DataFrame(train['cat3'].value_counts())
    count_cutoff = 250
    count_cat3 = count_cat3[count_cat3.cat3 >= count_cutoff]
    train['cat3'] = train.apply(lambda row: row['cat3'] if row['cat3'] in count_cat3.index else row['cat2'], axis=1)
    test_top['cat3'] = test_top.apply(lambda row: row['cat3'] if row['cat3'] in count_cat3.index else row['cat2'], axis=1)
    test_rest['cat3'] = test_rest.apply(lambda row: row['cat3'] if row['cat3'] in count_cat3.index else row['cat2'], axis=1)

    process_dataset(train, test_top, test_rest)
    cats = train['cat3'].unique().astype(str)
    nrow_train = train.shape[0]
    y = np.log1p(train['price'])
    merge: pd.DataFrame = pd.concat([train, test_top])
    merge: pd.DataFrame = pd.concat([train, test_top])
    merge['name'] = merge['name'].astype(str) + ' ' + merge['category_name'].astype(str) + ' ' + merge['brand_name']
    test_rest['new_name'] = test_rest['name'].astype(str) + ' ' + test_rest['category_name'].astype(str) + ' ' + test_rest['brand_name']
    tv = TfidfVectorizer(max_features=None,
                         ngram_range=(1, 3), min_df=2, token_pattern=r'(?u)\b\w+\b')
    X_name = tv.fit_transform(merge['name'])
    T_name = tv.transform(test_rest['new_name'])
    X_description = tv.fit_transform(merge['item_description'])
    T_description = tv.transform(test_rest['item_description'])
    X_category = tv.fit_transform(merge['category_name'])
    T_category = tv.transform(test_rest['category_name'])

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    T_brand = lb.transform(test_rest['brand_name'])
    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values)
    T_dummies = csr_matrix(pd.get_dummies(test_rest[['item_condition_id', 'shipping']], sparse=True).values)
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_name, X_category)).tocsr()
    sparse_test_rest = hstack((T_dummies, T_description, T_brand, T_name, T_category)).tocsr()
    X = sparse_merge[:nrow_train]
    X_test_top = sparse_merge[nrow_train:]
    cpuStats()

    del merge
    del sparse_merge
    del X_dummies
    del X_description
    del X_brand
    del X_name
    del T_dummies
    del T_description
    del T_brand
    del T_name
    gc.collect()
    model = Ridge(solver="sag", fit_intercept=True, random_state=369)
    model.fit(X, y)
    test_top_predsa['All_score'] = model.predict(X=X_test_top)
    test_rest_predsa['All_score'] = model.predict(X=sparse_test_rest)
    test_preds_all: pd.DataFrame = pd.concat([test_top_predsa, test_rest_predsa])
    print(test_preds_all.shape)
    cpuStats()
    all_cat_tt = pd.DataFrame(columns=['test_id','Ind_score'])
    all_cat_tr = pd.DataFrame(columns=['test_id','Ind_score'])
    
    cutoff = 50
    for cat in cats:
        print(cat)
        train_cat = train[train.cat3 == cat].reset_index(drop=True)
        if train_cat.shape[0] < cutoff:
            continue
        testtop_cat = test_top[test_top.cat3 == cat].reset_index(drop=True)
        testrest_cat = test_rest[test_rest.cat3 == cat].reset_index(drop=True)
        test_top_preds: pd.DataFrame = testtop_cat[['test_id']].reset_index(drop=True)
        test_rest_preds: pd.DataFrame = testrest_cat[['test_id']].reset_index(drop=True)
        nrow_cat = train_cat.shape[0]
        y = np.log1p(train_cat["price"])
        max_cat = y.max()
        min_cat = y.min()
        merge: pd.DataFrame = pd.concat([train_cat, testtop_cat])
        del train_cat
        del testtop_cat
        cv = CountVectorizer()
        X_category = cv.fit_transform(merge['category_name'])
        T_category = cv.transform(testrest_cat['category_name'])
        tv = TfidfVectorizer(max_features=None, ngram_range=(1, 3), min_df=2,token_pattern=r'(?u)\b\w+\b')
        merge['name'] = merge['name'] + ' ' + merge['brand_name']
        testrest_cat['name'] = testrest_cat['name'] + ' ' + testrest_cat['brand_name']
        X_name = tv.fit_transform(merge['name'])
        T_name = tv.transform(testrest_cat['name'])

        X_description = tv.fit_transform(merge['item_description'])
        T_description = tv.transform(testrest_cat['item_description'])
        lb = LabelBinarizer(sparse_output=True)
        X_brand = lb.fit_transform(merge['brand_name'])
        T_brand = lb.transform(testrest_cat['brand_name'])
        X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values)
        T_dummies = csr_matrix(pd.get_dummies(testrest_cat[['item_condition_id', 'shipping']], sparse=True).values)
        sparse_merge = hstack((X_dummies, X_description, X_brand, X_name, X_category)).tocsr()
        sparse_test_rest = hstack((T_dummies, T_description, T_brand, T_name, T_category)).tocsr()
        X = sparse_merge[:nrow_cat]

        X_test_top = sparse_merge[nrow_cat:]
        del merge
        del testrest_cat
        del sparse_merge
        del X_dummies
        del X_category
        del X_description
        del X_brand
        del X_name
        del T_dummies
        del T_category
        del T_description
        del T_brand
        del T_name
        gc.collect()
    
        model = Ridge(solver="saga", fit_intercept=False, random_state=369)
        model.fit(X, y)
        test_top_preds['Ind_score'] = model.predict(X=X_test_top)
        trpreds = model.predict(X=sparse_test_rest)
        test_rest_preds['Ind_score'] = trpreds
        test_top_preds['Ind_score'] = np.clip(test_top_preds['Ind_score'], min_cat, max_cat)
        test_rest_preds['Ind_score'] = np.clip(test_rest_preds['Ind_score'], min_cat, max_cat)
        all_cat_tt: pd.DataFrame = pd.concat([all_cat_tt, test_top_preds])
        all_cat_tr: pd.DataFrame = pd.concat([all_cat_tr, test_rest_preds])

    tt = test_top_predsa.merge(all_cat_tt,on='test_id',how='left')
    tr = test_rest_predsa.merge(all_cat_tr,on='test_id',how='left')
    tt['both'] = tt.apply(lambda row: row['All_score'] if pd.isnull(row['Ind_score']) else row['Ind_score'], axis=1)
    tr['both'] = tr.apply(lambda row: row['All_score'] if pd.isnull(row['Ind_score']) else row['Ind_score'], axis=1)
    print(tt['both'].head(180))
    preds_all: pd.DataFrame = pd.concat([tt, tr])
    preds_all['price'] = np.expm1((preds_all['both'] + preds_all['All_score']) / 2)
    preds_all[['test_id','price']].to_csv("All_ind_v23.csv",index=False)
    print(preds_all['All_score'].min())
    cpuStats()

    print(preds_all.head())
                         

if __name__ == '__main__':
    main()