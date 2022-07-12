# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import time
import gc
import re
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


# read raw data
start_time = time.time()
input_dir = os.path.join(os.pardir, 'input')
print('  Loading data...')
train_df = pd.read_json('../input/train.json') # store as dataframe objects
test_df = pd.read_json('../input/test.json')
print('    Time elapsed %.0f sec'%(time.time()-start_time))


# apply label encoding on the target variable
len_train = train_df.shape[0]
target = train_df['cuisine']
merged_df = pd.concat([train_df, test_df], sort=True)
lb = LabelEncoder()
target2 = lb.fit_transform(target)


# preprocess labels
print('  Pre-processing data...')
features_processed = []
for item in merged_df['ingredients']:
    newitem = []
    for ingr in item:
        ingr = ingr.lower() # to lowercase 
        ingr = re.sub("[^a-zA-Z]"," ",ingr) # remove punctuation, digits or special characters 
        ingr = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', ingr) # remove different units
        ingr = re.sub((r'\b(style|light|swanson|vegetarian|progresso|homemade|hellmann|hidden valley|johnsonville|wish bone|lipton|refrigerated|vegan|salted|original|mccormick|unsweetened|taco bell|medium|small|old el paso|unsalted)\b'), ' ', ingr) # remove some brand names
        ingr = re.sub("  "," ",ingr) # remove double space
        newitem.append(ingr)
    features_processed.append(newitem)
merged_df['ingredients'] = features_processed # put it back
print('    Time elapsed %.0f sec'%(time.time()-start_time))


# Text Data Features
print ("Prepare text data of Train and Test ... ")
d = merged_df['ingredients'][:len_train]
train_text = [" ".join(doc) for doc in d]
d = merged_df['ingredients'][len_train:]
test_text = [" ".join(doc) for doc in d]
del d
gc.collect()


# Feature Engineering
print ("TF-IDF on text data ... ")
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(binary=True)
X = tfidf.fit_transform(train_text).astype(np.float32)
X_test = tfidf.transform(test_text).astype(np.float32)
del train_text, test_text
gc.collect()


# use lightgbm for regression
print(' start training...\n    Time elapsed %.0f sec'%(time.time()-start_time))
# specify config as a dict
params = {
    'max_depth': 8,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 20,
    'metric': 'multi_error',
    'num_leaves': 7,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'verbose': -1
}


# do the training
num_folds = 5
test_x = X_test
oof_preds = np.zeros([len_train])
sub_preds = np.zeros([test_x.shape[0],20])
folds = KFold(n_splits=num_folds, shuffle=True, random_state=4564)
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X)):
    lgb_train = lgb.Dataset(X[train_idx], target2[train_idx])
    lgb_valid = lgb.Dataset(X[valid_idx], target2[valid_idx])
        
    # train
    gbm = lgb.train(params, lgb_train, 1000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=20, verbose_eval=100)
    pr1 = gbm.predict(X[valid_idx], num_iteration=gbm.best_iteration)
    pr2 = gbm.predict(test_x, num_iteration=gbm.best_iteration)
    oof_preds[valid_idx] = pr1.argmax(axis=1)
    sub_preds += pr2 / folds.n_splits
    valid_idx += 1
sub_preds = sub_preds.argmax(axis=1)
e = (target2==oof_preds).mean()
print('Full validation score/error %.4f/%.4f' %(e,1-e))
print('    Time elapsed %.0f sec'%(time.time()-start_time))


# Write submission file
out_df = pd.DataFrame({'id': merged_df['id'][len_train:]})
pred2 = lb.inverse_transform(sub_preds)
out_df['cuisine'] = pred2
out_df.to_csv('submission.csv', index=False)
print('    Time elapsed %.0f sec'%(time.time()-start_time))
