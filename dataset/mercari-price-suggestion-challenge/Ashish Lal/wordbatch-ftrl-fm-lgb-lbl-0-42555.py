# Based on Bojan -> https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944
# and Nishant -> https://www.kaggle.com/nishkgp/more-improved-ridge-2-lgbm

import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.linear_model import SGDRegressor

import os
import sys
import psutil

###Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
###until Kaggle admins fix the wordbatch pip package installation
sys.path.insert(0, '../input/workbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL, NN_ReLU_H2

from nltk.corpus import stopwords
import re

NUM_BRANDS = 4500
NUM_CATEGORIES = 1200

# develop = False
develop= True

def print_memory_usage():
    print('cpu: {}'.format(psutil.cpu_percent()))
    print('consuming {:.2f}GB RAM'.format(
    	   psutil.Process(os.getpid()).memory_info().rss / 1073741824),
    	  flush=True)

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


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


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])


# def main():
start_time = time.time()
from time import gmtime, strftime
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# if 1 == 1:
train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv', engine='c')
test = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv', engine='c')

#train = pd.read_table('../input/train.tsv', engine='c')
#test = pd.read_table('../input/test.tsv', engine='c')

print('[{}] Finished to load data'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)
nrow_test = train.shape[0]  # -dftt.shape[0]
dftt = train[(train.price < 1.0)]
train = train.drop(train[(train.price < 1.0)].index)
del dftt['price']
nrow_train = train.shape[0]
# print(nrow_train, nrow_test)
y = np.log1p(train["price"])

print(train.shape)
print('5 folds scaling the test_df')
print(test.shape)
test_len = test.shape[0]
def simulate_test(test):
    if test.shape[0] < 800000:
        indices = np.random.choice(test.index.values, 2800000)
        test_ = pd.concat([test, test.iloc[indices]], axis=0)
        return test_.copy()
    else:
        return test

submission: pd.DataFrame = test[['test_id']]
print(len(submission))
test = simulate_test(test)
print('new shape ', test.shape)
print('[{}] Finished scaling test set...'.format(time.time() - start_time))
print_memory_usage()

merge: pd.DataFrame = pd.concat([train, dftt, test])


del train
del test
gc.collect()

merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
    zip(*merge['category_name'].apply(lambda x: split_cat(x)))
merge.drop('category_name', axis=1, inplace=True)
print('[{}] Split categories completed.'.format(time.time() - start_time))
print_memory_usage()

handle_missing_inplace(merge)
print('[{}] Handle missing completed.'.format(time.time() - start_time))
print_memory_usage()

cutting(merge)
print('[{}] Cut completed.'.format(time.time() - start_time))
print_memory_usage()

to_categorical(merge)
print('[{}] Convert categorical completed'.format(time.time() - start_time))
print_memory_usage()

# wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [-2.65, -0.35],
#                                                               "hash_size": 2 ** 22, "norm": None, "tf": 'binary',
#                                                               "idf": 2.0,
#                                                               }), procs=8)

# wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [-2.85, -0.35],
#                                                               "hash_size": 2 ** 22, "norm": l2, "tf": 1.0,
#                                                               "idf": 4.0,
#                                                               }), procs=8)
# wb.dictionary_freeze= True
# X_name = wb.fit_transform(merge['name'])


        
# del(wb)
# X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

cv_name = CountVectorizer(min_df=2, ngram_range=(1, 1),
                                       binary=True, token_pattern="\w+")
X_name = 2 * cv_name.fit_transform(merge['name'])
cv_name2 = CountVectorizer(min_df=2, ngram_range=(2, 2),
                                binary=True, token_pattern="\w+")
X_name2 = 0.5 * cv_name2.fit_transform(merge['name'])

print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))
print_memory_usage()

wb = CountVectorizer(min_df=2)
X_category1 = wb.fit_transform(merge['general_cat'])
X_category2 = wb.fit_transform(merge['subcat_1'])
X_category3 = wb.fit_transform(merge['subcat_2'])
print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))
print_memory_usage()

# wb= wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 3, "hash_ngrams_weights": [1.0, 1.0, 0.5],
# wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [-2.0, -1.55],
#                                                               "hash_size": 2 ** 22, "norm": "l2", "tf": 1.0,
#                                                               "idf": 2.0})
                                                             
#                          , procs=8)
wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [-2.20, -1.55],
                                                              "hash_size": 2 ** 22, "norm": "l2", "tf": 1.0,
                                                              "idf": 4.0})
                                                             
                         , procs=8)
# FM_FTRL dev RMSLE: 0.423859971676
wb.dictionary_freeze= True
X_description = wb.fit_transform(merge['item_description'])
del(wb)
X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))
print_memory_usage()

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))
print_memory_usage()

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                      sparse=True).values)
print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
print_memory_usage()

print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
      X_name.shape)
sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name, X_name2)).tocsr()

print('[{}] Create sparse merge completed'.format(time.time() - start_time))
del X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name
del merge
gc.collect()
print_memory_usage()

#    pd.to_pickle((sparse_merge, y), "xy.pkl")
# else:
#    nrow_train, nrow_test= 1481661, 1482535
#    sparse_merge, y = pd.read_pickle("xy.pkl")

# Remove features with document frequency <=1
print(sparse_merge.shape)
mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
sparse_merge = sparse_merge[:, mask]
X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_test:]
print(sparse_merge.shape)
print_memory_usage()

D_1 = sparse_merge.shape[1]
D_0 = sparse_merge.shape[0]
del sparse_merge
gc.collect()
train_X, train_y = X, y
if develop:
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.01, random_state=100)
print_memory_usage()

predictions = []

model = FTRL(alpha=0.01, beta=0.1, L1=0.00002, L2=1.0, D=D_1, iters=50, inv_link="identity", threads=1)
model.fit(train_X, train_y)
print('[{}] Train FTRL completed'.format(time.time() - start_time))
if develop:
    preds = model.predict(X=valid_X)
    print("FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
    predictions.append(preds)
    del preds
predsF = model.predict(X_test)

# model = FTRL(alpha=0.01, beta=0.1, L1=0.00002, L2=1.0, D=D_1, iters=100, inv_link="identity", threads=1)
# model.fit(train_X, train_y)
# print('[{}] Train FTRL completed'.format(time.time() - start_time))
# if develop:
#     preds = model.predict(X=valid_X)
#     print("FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
#     predictions.append(preds)
#     del preds
# predsF = model.predict(X_test)


# model = Ridge(solver="sag", fit_intercept=True, alpha = 4.5, random_state=666)

# model.fit(train_X, train_y)
# print('[{}] Train Ridge completed'.format(time.time() - start_time))
# if develop:
#     preds = model.predict(X=valid_X)
#     print("Ridge dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
#     predictions.append(preds)
#     del preds

# predsR = model.predict(X_test)
# print('[{}] Predict Ridge completed'.format(time.time() - start_time))
# print_memory_usage()


# model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, \
#                 D=D_1, alpha_fm=0.01, L2_fm=0.0, init_fm=0.01, \
#                 D_fm=200, e_noise=0.0001, iters=15, inv_link="identity", threads=2)

# # FM_FTRL dev RMSLE: 0.4283449138

# model.fit(train_X, train_y)
# print('[{}] Train FM FTRL completed'.format(time.time() - start_time))
# print_memory_usage()

# if develop:
#     preds = model.predict(X=valid_X)
#     print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
#     predictions.append(preds)
#     del preds

# predsFM = model.predict(X_test)
# print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))

model = FM_FTRL(alpha=0.02, beta=0.02, L1=0.00002, L2=0.1, \
                D=D_1, alpha_fm=0.01, L2_fm=0.002, init_fm=0.01, \
                D_fm=200, e_noise=0.0001, iters=15, inv_link="identity", threads=2)

# FM_FTRL dev RMSLE: 0.428139925721, alpha=0.02
# FM_FTRL dev RMSLE: 0.428078275240844, alpha=0.02, beta=0.02
# FM_FTRL dev RMSLE: 0.42807364081975635,  alpha=0.02, beta=0.02, L1 = 0.00002
model.fit(train_X, train_y)
print('[{}] Train FM FTRL completed'.format(time.time() - start_time))
print_memory_usage()

if develop:
    preds = model.predict(X=valid_X)
    print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
    predictions.append(preds)
    del preds

predsFM = model.predict(X_test)
print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))


# del model, train_X, train_y
del model
print_memory_usage()

# print('[{}] Training NN...'.format(time.time() - start_time))
# model = NN_ReLU_H2(alpha=0.1,L2=0.001,
# 			   	 e_noise=0.0001,
# 				 D=D_1,
# 				 D_nn=512,
# 				 D_nn2=1,
# 				 init_nn=0.0001,
# 				 iters=3,
# 				 inv_link= "linear",
# 				 threads= 1,
# 				 seed= 0)

# # NN dev RMSLE: 0.46296838464997764, alpha=0.2			
# # NN dev RMSLE: 0.4468662841922423, alpha = 0.1
# model.fit(train_X, train_y)
# print('[{}] Train NN completed'.format(time.time() - start_time))
# print_memory_usage()

# if develop:
#     preds = model.predict(X=valid_X)
#     print("NN dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
#     predictions.append(preds)
#     del preds

# predsN = model.predict(X_test)
# print('[{}] Predict NN completed'.format(time.time() - start_time))

# from scipy.optimize import minimize


# if develop:
#     starting_values = [0.5]*len(predictions)  #adding constraints  and a different solver 
#     cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#     #our weights are bound between 0 and 1
#     bounds = [(0,1)]*len(predictions)  
    
#     def rmsle_func(weights):
#         ''' scipy minimize will pass the weights as a numpy array '''
#         final_prediction = 0
#         for weight, prediction in zip(weights, predictions):
#             final_prediction += weight*prediction

#         return rmsle(valid_y, final_prediction)  
    
#     res = minimize(rmsle_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

#     print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
#     print('Best Weights: {weights}'.format(weights=res['x']))

params = {
    'learning_rate': 0.75,
    'application': 'regression',
    'max_depth': 4,
    'num_leaves': 100,
    'verbosity': -1,
    'metric': 'RMSE',
    'data_random_seed': 1,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'feature_fraction': 0.6,
    'nthread': 2,
    'min_data_in_leaf': 100,
    'max_bin': 8192
}

# params = {
#     'learning_rate': 0.75,
#     'application': 'regression',
#     'max_depth': 3,
#     'num_leaves': 100,
#     'verbosity': -1,
#     'metric': 'RMSE',
#     'max_bin': 8192,
#     'nthread': 2
# }

# Remove features with document frequency <=100
# print(sparse_merge.shape)
# mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 100, 0, 1), dtype=bool)
# sparse_merge = sparse_merge[:, mask]
# X = sparse_merge[:nrow_train]
# X_test = sparse_merge[nrow_test:]
# print(sparse_merge.shape)

# develop=True
# train_X, train_y = X, y
# if develop:
#     train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)

# d_train = lgb.Dataset(train_X, label=train_y)
# watchlist = [d_train]
# if develop:
#     d_valid = lgb.Dataset(valid_X, label=valid_y)
#     watchlist = [d_train, d_valid]

# model = lgb.train(params, train_set=d_train, num_boost_round=6000, valid_sets=watchlist, \
#                   early_stopping_rounds=1000, verbose_eval=1000)

# if develop:
#     preds = model.predict(valid_X)
#     print("LGB dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

# predsL = model.predict(X_test)

# print('[{}] Predict LGB completed.'.format(time.time() - start_time))
# print_memory_usage()

# preds = (predsF * 0.2 + predsL * 0.3 + predsFM * 0.5)

preds = (predsF * 0.02 +  predsFM * 0.98)
 
preds = preds[:test_len]
print(len(preds))
print(len(submission))
submission['price'] = np.expm1(preds)
print_memory_usage()
submission.to_csv("submission_wordbatch_ftrl_fm_.csv", index=False)


# if __name__ == '__main__':
    # main()