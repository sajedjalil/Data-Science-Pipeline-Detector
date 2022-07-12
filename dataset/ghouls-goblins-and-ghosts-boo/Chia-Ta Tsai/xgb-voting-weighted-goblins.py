import time
from itertools import combinations
from math import sqrt, log1p, expm1
import numpy as np
import pandas as pd
from random import randint, random, seed, uniform, choice
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

target, target_id = 'type', 'id' 

def create_submission(output_id, output_val, prefix='', compression=False):

    filename = 'submission_{}_{}.csv'.format(prefix, time.strftime("%Y-%m-%d-%H-%M"))
    
    output_id = output_id.astype(int)
    submission = pd.DataFrame(data={target_id: output_id, target: output_val})
    #a.to_frame().join(b.to_frame())
    if compression:
        filename += '.gz'
        print('\nMake submission:{}\n'.format(filename))
        submission.to_csv(filename, index=False, header=True, compression='gzip')
    else:
        print('\nMake submission:{}\n'.format(filename))
        submission.to_csv(filename, index=False, header=True)


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#weighting
weight = train[target].replace({'Goblin':1.15, 'Ghost':1.0, 'Ghoul':1.0})
#
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(train[target].values)
train.drop([target_id, target], axis=1, inplace=True)
test_id = test[target_id]
test.drop([target_id], axis=1, inplace=True)



df = pd.concat([train, test])
list_feat_num = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']
for i, j in combinations(list_feat_num, 2):
    str1 = 'sqrt{}*{}'.format(i, j)
    df[str1] = df[i] * df[j]
    df[str1] = df[str1].apply(sqrt)
    str1 = 'log1p{}-{}'.format(i, j)
    df[str1] = df[i].apply(log1p) - df[j].apply(log1p)
    str1 = 'expm1{}+{}'.format(i, j)
    df[str1] = df[i].apply(expm1) + df[j].apply(expm1)
df = pd.get_dummies(df, columns=['color'])

#xgb
xgb_param = {'silent':1, 'objective':'multi:softmax', 'num_class':len(set(y_encoded)), 'eval_metric':'mlogloss', 'nthread':8}
xgb_param['eta'] = 0.025

#split
train, test = df[:train.shape[0]], df[train.shape[0]:]
dtest = xgb.DMatrix(test)

collection = []

nr_split = 100
nr_round, nr_min_round = 1000000, 50
va_ratio, tr_ratio = 0.25, 0.50

random_state=1014712
seed(random_state)

ss = StratifiedShuffleSplit(n_splits=nr_split, test_size=va_ratio, train_size=tr_ratio, random_state=random_state)
#ss = ShuffleSplit(n_splits=nr_split, test_size=va_ratio, train_size=tr_ratio, random_state=random_state)
for ind_tr, ind_va in ss.split(train, y=y_encoded): 
    y_train, y_valid = y_encoded[ind_tr], y_encoded[ind_va]
    weight_tr = weight[ind_tr]
    dtrain = xgb.DMatrix(train.iloc[ind_tr], label=y_train, weight=weight_tr)
    dvalid = xgb.DMatrix(train.iloc[ind_va], label=y_valid) 

    #rand param shake up
    xgb_param['max_depth'] = randint(8, 16)
    xgb_param['colsample_bytree'] = uniform(0.25, 0.50)
    xgb_param['subsample'] = uniform(0.25, 0.75)
    xgb_param['gamma'] = 2 ** uniform(-5.0, 5.0)
    xgb_param['min_child_weight'] = 2.0 ** uniform(0.0, 5.0) #default=1
    xgb_param['lambda'] = 2 ** uniform(0.0, 5.0) #tree=1, linear=0
    xgb_param['seed'] = int(65536 * random())
    
    watchlist  = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(xgb_param, dtrain, nr_round, 
                    evals=watchlist, 
                    early_stopping_rounds=nr_min_round)
    
    y_pred = gbm.predict(dvalid, ntree_limit=gbm.best_ntree_limit).flatten()
    y_pred = [int(i) for i in y_pred]
    acc = accuracy_score(y_valid, y_pred)
    
    test_pred = gbm.predict(dtest, ntree_limit=gbm.best_ntree_limit).flatten()
    test_pred = [int(i) for i in test_pred]
    test_pred = encoder.inverse_transform(test_pred)
    create_submission(test_id, pd.Series(test_pred), prefix='s{:.4f}'.format(acc), compression=False)
    
    if acc > 0.65 and acc < 0.80:
        collection.append(test_pred) 

test_pred = []
collection = pd.DataFrame(collection)
collection.drop_duplicates(inplace=True)
for col in collection.columns.tolist():
    freq = collection[col].value_counts()
    test_pred.append(freq.index[0])
create_submission(test_id, test_pred, prefix='major{}-{}'.format(nr_split, len(collection)), compression=False)
    

