import numpy as np
import pandas as pd
import os
import time
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import lightgbm as lgb
import scipy


# read raw data
print(os.listdir('../input'))
start_time = time.time()
input_dir = os.path.join(os.pardir, 'input')
print('  Loading data...')
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
len_train = train_df.shape[0]
target = train_df['target']
merged_df = pd.concat([train_df, test_df], sort=True)
train_text = merged_df['question_text'][:len_train]
test_text = merged_df['question_text'][len_train:]


# Feature Engineering
print ('TF-IDF on text data ... ')
print('    Time elapsed %.0f sec'%(time.time()-start_time))
tfidf = TfidfVectorizer(
    max_features=100000
    ,ngram_range=(1, 2)
    #,stop_words='english'   # this makes it way worse
    #,binary=True            # this makes it worse
    )
X = tfidf.fit_transform(train_text).astype(np.float32)
print( X.shape )
X_test = tfidf.transform(test_text).astype(np.float32)
del train_text, test_text
gc.collect()


# add some columns
# number of characters
print('add some columns')
print('    Time elapsed %.0f sec'%(time.time()-start_time))
a=np.array([len(x) for x in merged_df['question_text']]).reshape(merged_df.shape[0],1)
X = scipy.sparse.hstack((a[:len_train,:],X), format='csr', dtype='float32')
X_test = scipy.sparse.hstack((a[len_train:,:],X_test), format='csr', dtype='float32')
# number of words
a=np.array([len(x.split()) for x in merged_df['question_text']]).reshape(merged_df.shape[0],1)
X = scipy.sparse.hstack((a[:len_train,:],X), format='csr', dtype='float32')
X_test = scipy.sparse.hstack((a[len_train:,:],X_test), format='csr', dtype='float32')
# number of title case words
a=np.array(merged_df['question_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))).reshape(merged_df.shape[0],1)    
X = scipy.sparse.hstack((a[:len_train,:],X), format='csr', dtype='float32')
X_test = scipy.sparse.hstack((a[len_train:,:],X_test), format='csr', dtype='float32')
# number of upper case words
a=np.array(merged_df['question_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))).reshape(merged_df.shape[0],1)    
X = scipy.sparse.hstack((a[:len_train,:],X), format='csr', dtype='float32')
X_test = scipy.sparse.hstack((a[len_train:,:],X_test), format='csr', dtype='float32')
print( X.shape )


# use lightgbm for regression
print(' start training...\n    Time elapsed %.0f sec'%(time.time()-start_time))
# specify config as a dict
params = {
    'max_depth': 8,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}


# do the training
num_folds = 5
test_x = X_test
oof_preds = np.zeros([len_train])
sub_preds = np.zeros([test_x.shape[0]])
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=4564)
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, target)):
    lgb_train = lgb.Dataset(X[train_idx], target[train_idx])
    lgb_valid = lgb.Dataset(X[valid_idx], target[valid_idx])
        
    # train
    gbm = lgb.train(params, lgb_train, 5000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=100, verbose_eval=200)
    oof_preds[valid_idx] = gbm.predict(X[valid_idx], num_iteration=gbm.best_iteration)
    sub_preds += gbm.predict(test_x, num_iteration=gbm.best_iteration) / folds.n_splits


m0 = 0
for t in range(4, 50, 1):
    m = metrics.f1_score(target, (oof_preds>t/100).astype(int))
    if m > m0:
        m0 = m
        t0 = t / 100
        print( t/100, m )
e = metrics.f1_score(target, (oof_preds>t0).astype(int))
print('Full validation F1/t0 %.4f/%.2f' %(e, t0))


# Write submission file
out_df = pd.DataFrame({'qid': merged_df['qid'][len_train:]})
out_df['prediction'] = (sub_preds>t0).astype(int)
out_df.to_csv('submission.csv', index=False)
print('    Time elapsed %.0f sec'%(time.time()-start_time))