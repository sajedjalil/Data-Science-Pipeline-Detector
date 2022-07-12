# -*- coding: utf-8 -*-
"""
Created on Tue Mar  20 01:08:25 2018

thanks a lot to Indian Decision Scientist @Jagan

@author: pengb
"""
import os
print(os.listdir('../input'))

import warnings
warnings.filterwarnings("ignore")

import gc
import time
start_time = time.time()

import pandas as pd

print('===========================================================================\n')
print( '从磁盘读取数据...' + '\t' + 'Reading data from disk...')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.info())
print('===========================================================================\n')
print(test.info())

y = train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']]
train.drop(['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate'], inplace=True, axis=1)

rows_train=train.shape[0]
rows_test=test.shape[0]
train_id=train['id']
test_id=test['id']

merge = pd.concat([train, test], axis=0)

from nltk.corpus import stopwords
import gensim
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
eng_stopwords = set(stopwords.words("english"))

def clean(text_col):

    word_list = gensim.utils.simple_preprocess(text_col, deacc=True)
    # Phrase功能： new + york --> new_york
    bigram = gensim.models.Phrases(text_col)
    clean_words = [w for w in word_list if not w in eng_stopwords]
    clean_words = bigram[clean_words]
    # verb化,重点将衍生词转化为verb
    clean_words=[lem.lemmatize(word, "v") for word in clean_words]
    return(' '.join(clean_words))  

print('===========================================================================\n')
print('清洗之前:'  + '\t' +  'Before cleaning:' + '\n',merge.comment_text.iloc[2018])
print('清洗之后:' + '\t' +  'After cleaning:' + '\n',clean(merge.comment_text.iloc[2018]))
merge['comment_text'] = merge['comment_text'].apply(clean)

from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
merge_vectorized = cv.fit_transform(merge.comment_text.values)
train_vectorized = merge_vectorized[0:rows_train]
test_vectorized = merge_vectorized[rows_train:]


del train
del test
del merge['id']
gc.collect()


from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

train_preds = pd.DataFrame()
test_preds = pd.DataFrame()

for col in y.columns:
    X_train, X_test, y_train, y_test = train_test_split(train_vectorized, y[col], test_size=0.7, random_state=64)

############
############
##LIGHTGBM##
############
############


    params = {
        'boosting_type': 'gbdt',    #采取梯度提升
        'objective': 'binary',    #application 选择回归方式
        'metric': 'auc',    # params['metric'] = 'l1'    # l1惩罚
        'max_depth': 16,    
        'num_leaves': 31,    # 一棵树上的叶子数
        'learning_rate': 0.25,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,    # 在不进行重采样的情况下随机选择部分数据
        'bagging_freq': 5,
        'verbose': 1,    #=0 = 错误 (警告)
        'num_threads': 4,  # 线程数，与cpu核数有关，一核两线程
        'lambda_l2': 1,    # L1正则项
        'min_gain_to_split': 0,    # 执行切分的最小增益
        'seed':1234,
        'min_data': 28,    # 一个叶子上数据的最小数量，避免过拟合
        'min_hessian': 0.05    # 一个叶子上的最小 hessian 和,避免过拟合
        }  



    model = lgb.train(
            params,
            lgb.Dataset(X_train, y_train),
            num_boost_round=10000,
            valid_sets=[lgb.Dataset(X_test, y_test)],
            early_stopping_rounds=100,
            verbose_eval=25)

    train_preds[col] = model.predict(X_test, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_test, train_preds[col])
    print('AUC:', auc)
    test_preds[col] = model.predict(test_vectorized, num_iteration=model.best_iteration)

end_time=time.time()
print('===========================================================================\n')
print('total time in finishing LGB model...',round(end_time-start_time, 2),'s')


############
############
##XGBOOST###
############
############


train_preds_ = pd.DataFrame()
test_preds_ = pd.DataFrame()

for col in y.columns:
    X_train, X_test, y_train, y_test = train_test_split(train_vectorized, y[col], test_size=0.7, random_state=64)
    
    xgb_params = {
            'eta': 0.037,    # 防止过拟合,boosting过程收缩步长
            'max_depth': 5,    # 树的最大深度
            'subsample': 0.80,    # 随机抽取80%原始数据子样本建立树模型
            'objective': 'reg:linear',
            'eval_metric': 'auc',    # 平均绝对误差
            'lambda': 0.8,    # L2正则惩罚系数
            'alpha': 0.4,     # L1正则惩罚系数，高纬度数据使用
            'silent': 1   # boosting过程不输出信息
            }
    df_train = xgb.DMatrix(X_train, y_train)
    df_test = xgb.DMatrix(X_test, y_test)
    d_test = xgb.DMatrix(test_vectorized)
    watchlist = [(df_train, 'train'), (df_test, 'test')]
    model_xgb = xgb.train(xgb_params, df_train, 500, watchlist, verbose_eval=50, early_stopping_rounds=20)



    train_preds_[col]= model_xgb.predict(df_test)  
    test_preds_[col] = model_xgb.predict(d_test)  

    auc = roc_auc_score(y_test, train_preds_[col])
    print('AUC:',auc)

end_time=time.time()
print('===========================================================================\n')
print('total time in finishing XGBoost model...',round(end_time-start_time, 2),'s')


"""
    the results show that lgb model perform better than xgb model.
    so, only lgb model will be uesd.
"""


submission = pd.DataFrame()
submission['id'] = test_id
for col in test_preds.columns:
    submission[col] = test_preds[col]

from datetime import datetime
print('===========================================================================\n')
print( '将结果写入磁盘...' + '\t' + 'Writing the results to disk...')
submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
print( '\n完成作业...' + '\t' + 'Done...')
end_time=time.time()
print('共耗时', round(end_time-start_time, 2), 's' + '\t' + 'It took a total of', round(end_time-start_time, 2), 's')