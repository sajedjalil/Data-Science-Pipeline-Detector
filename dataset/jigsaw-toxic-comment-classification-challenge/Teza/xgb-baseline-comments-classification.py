# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc
from sklearn.feature_extraction.text import TfidfVectorizer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train = train.fillna("unknown")
test = test.fillna("unknown")


train_mes, valid_mes, train_l, valid_l = train_test_split(train['comment_text'],train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']], test_size=0.2, random_state=2)


def text_process(comment):
    nopunc = [char for char in comment if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#Couldnt remove the stop words using the above function since it is taking too long
#Can try it on a local machine, I feel it improves the score-Not sure though


'''
transform_com = CountVectorizer().fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))
comments_train = transform_com.transform(train['comment_text'])
comments_test = transform_com.transform(test['comment_text'])
gc.collect()'''


transform_com = TfidfVectorizer().fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))
'''comments_train = transform_com.transform(train['comment_text'])'''
comments_train = transform_com.transform(train_mes)
comments_valid = transform_com.transform(valid_mes)
comments_test = transform_com.transform(test['comment_text'])
gc.collect()



import xgboost as xgb
'''
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2, num_rounds=400):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.1
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'logloss'
    param['min_child_weight'] = 1
    param['subsample'] = 0.4
    param['colsample_bytree'] = 0.6
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    xgtest = xgb.DMatrix(test_X)
    model = xgb.train(plst, xgtrain, num_rounds)
    pred_test_y = model.predict(xgtest)
    return pred_test_y, model
'''


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2017, num_rounds=1000):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.1
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'logloss'
    param['min_child_weight'] = 1
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return model
    

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((test.shape[0], len(col)))

for i, j in enumerate(col):
    print('fit '+j)
    model = runXGB(comments_train, train[j], comments_valid,comments_valid[j])
    preds[:,i] = model.predict(xgb.DMatrix(comments_text))
    gc.collect()

subm = pd.read_csv('../input/sample_submission.csv')    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('xgb.csv', index=False)
    

# Any results you write to the current directory are saved as output.