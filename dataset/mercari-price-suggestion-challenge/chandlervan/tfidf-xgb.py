# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time

start = time.time()
PATT1 = re.compile(r'\[rm\]')
PATT2 = re.compile('[a-z| |\'s|]+')
def letter_clean(x):
    x = str(x).lower()
    x = re.sub(PATT1,'',x)
    x = PATT2.search(x)
    if x == None or x == 'missing':
        return 'no description yet'
    else:
        return x.group().strip()

def categroy_split(dataset):
    dataset['category1'] = dataset['category_name'].map(lambda x:x.split('/')[0])
    dataset['category2'] = dataset['category_name'].map(lambda x:x.split('/')[1])
    return dataset

def non_letter(x):
    x = str(x).lower()
    x = PATT2.search(x)
    if x == None:
        return 0
    else:
        return len(x.group().strip())

train = pd.read_csv('../input/train.tsv',delimiter = '\t')
test = pd.read_csv('../input/test.tsv',delimiter = '\t')
#处理categroy_name
train['category_name'] = train['category_name'].fillna('missing/missing')
test['category_name'] = test['category_name'].fillna('missing/missing')
train = categroy_split(train)
test = categroy_split(test)
del train['category_name']
del test['category_name']
gc.collect()
print('finishi category_name processing')
print(time.time()-start)
#处理brand_name
train['brand_name'] = train['brand_name'].fillna('missing')
test['brand_name'] = test['brand_name'].fillna('missing')
le = LabelEncoder()
le.fit(np.hstack([train.brand_name,test.brand_name]))
train.brand_name = le.transform(train.brand_name)
test.brand_name = le.transform(test.brand_name)
print('finishi brand_name processing')
print(time.time()-start)
#处理category1 和 category2
le1 = LabelEncoder()
le1.fit(np.hstack([train.category1,test.category1]))
train.category1 = le1.transform(train.category1)
test.category1 = le1.transform(test.category1)
le2 = LabelEncoder()
le2.fit(np.hstack([train.category2,test.category2]))
train.category2 = le2.transform(train.category2)
test.category2 = le2.transform(test.category2)
print('finishi cat1 cat2 processing')
print(time.time()-start)
#第一次处理name字段
train['name'] = train['name'].fillna('missing')
test['name'] = test['name'].fillna('missing')
train.name = train.name.map(letter_clean)
test.name = test.name.map(letter_clean)
print('finishi first name processing')
print(time.time()-start)
#第一次处理item_description字段
train['item_description'] = train['item_description'].fillna('missing')
test['item_description'] = test['item_description'].fillna('missing')
train.item_description = train.item_description.map(letter_clean)
test.item_description = test.item_description.map(letter_clean)
print('finishi first desc processing')
print(time.time()-start)
#将name 和 item_description 转化为tfidf
tv = TfidfVectorizer()
tv_name = tv.fit_transform(pd.concat([train['name'],test['name']]))
tv = TfidfVectorizer()
tv_description = tv.fit_transform(pd.concat([train['item_description'],test['item_description']]))
print('finishi tfidf processing')
print(time.time()-start)
#处理dummies数据
cat1_dummies = pd.get_dummies(pd.concat([train['category1'],test['category1']]))
cat2_dummies = pd.get_dummies(pd.concat([train['category2'],test['category2']]))
condition_dummies = pd.get_dummies(pd.concat([train['item_condition_id'],test['item_condition_id']]))
csr_dummise = hstack((cat1_dummies,cat2_dummies,condition_dummies)).tocsr()
print('finishi dummies processing')
print(time.time()-start)
#对item_description 和 name做一些统计信息
train['description_cnt'] = train.item_description.map(lambda x:len(str(x)))
train['name_cnt'] = train.name.map(lambda x:len(str(x)))
test['description_cnt'] = train.item_description.map(lambda x:len(str(x)))
test['name_cnt'] = train.name.map(lambda x:len(str(x)))
train['description_letter'] = train['item_description'].map(non_letter)
train['name_letter'] = train['name'].map(non_letter)
test['description_letter'] = test['item_description'].map(non_letter)
test['name_letter'] = test['name'].map(non_letter)
np_price = np.log1p(train['price'])
print('finishi stastic processing')
print(time.time()-start)
#处理完毕，开始删除字段清除内存
del train['price']
del train['item_description']
del train['name']
del test['item_description']
del test['name']
del train['train_id']
del test['test_id']
del train['category1']
del train['category2']
del train['item_condition_id']
del test['category1']
del test['category2']
del test['item_condition_id']
gc.collect()
print('finishi del processing')
print(time.time()-start)
#构造训练集和测试集
csr_train = hstack((train,tv_name[:len(train)],tv_description[:len(train)],csr_dummise[:len(train)])).tocsr()
csr_test = hstack((test,tv_name[len(train):],tv_description[len(train):],csr_dummise[len(train):])).tocsr()
del train
del test
del tv_name
del tv_description
del csr_dummise
gc.collect()
#划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(csr_train,np_price,test_size = 0.1,random_state=2017)
# #使用xgb进行训练
xgb_params = {
    'objective': 'reg:linear',
    'eta': 0.02,
    'max_depth': 10,
    'learning_rate': 0.4,
    'seed': 0,
    'silent': 1,
}
xgbtrain_tr = xgb.DMatrix(X_train, y_train.astype('float'))
xgbtest_va = xgb.DMatrix(X_train, y_train.astype('float'))
xgbtest = xgb.DMatrix(X_test)
xgbtest_real = xgb.DMatrix(csr_test)
watchlist = [(xgbtrain_tr, 'train'), (xgbtest_va, 'test')]
model_xgb = xgb.train(xgb_params, xgbtrain_tr,360,early_stopping_rounds=15,evals=watchlist)
test_re = model_xgb.predict(xgbtest)
print('finish training')
score_te = 0
def squared_log_error(pred, actual):
    return (pred-actual)**2
for vv in range(len(test_re)):
    score_te += squared_log_error(y_test.values[vv],test_re[vv])
print(np.sqrt(score_te/len(y_test)))

real = model_xgb.predict(xgbtest_real)
sub = pd.DataFrame(np.expm1(real)).reset_index()
sub.columns = ['test_id','price']
sub.to_csv('sample_submission.csv',index=False)