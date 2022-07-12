import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import xgboost as xgb

from sklearn.model_selection import cross_val_score

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_y = train_df['loss']
test_id = test_df['id']

train_df.drop('loss',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)
train_df.drop('id',axis=1,inplace=True)

to_drop = []

train_df.drop(to_drop,axis=1,inplace=True)
test_df.drop(to_drop,axis=1,inplace=True)

features = train_df.columns
cat_feat = []
cont_feat = []

for feature in features:
    if feature.startswith('cat'):
        cat_feat.append(feature)
    else:
        cont_feat.append(feature)

for feature in cat_feat:
    enc = LabelEncoder()
    enc.fit(pd.concat([train_df[feature],test_df[feature]],ignore_index=True))
    train_df[feature] = enc.transform(train_df[feature])
    test_df[feature] = enc.transform(test_df[feature])

train_X = train_df.as_matrix()
train_Y = train_y.as_matrix()
test_X = test_df.as_matrix()



#clf = RandomForestRegressor(n_estimators=100,verbose=1)
#scores = cross_val_score(clf,train_X, train_Y, cv=5)
#print scores

to_submit = True
if to_submit:
    X_train, X_val, y_train, y_val = cross_validation.train_test_split(train_df, train_y, train_size=.90)
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_val, y_val)

    params = {
        "objective": "reg:linear",
        "booster": "gbtree",
        "eta": 0.05,
        "silent": 1,
        "max_depth": 8,
    }
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, 300, evals=watchlist, verbose_eval=True)

    pred = gbm.predict(xgb.DMatrix(test_df))
    _submit = open('submit_xgb.csv','w')
    _submit.write('id,loss\n')
    for i in range(0,len(pred)):
        _submit.write(str(test_id[i])+","+str(pred[i])+"\n")

    _submit.flush()
    _submit.close()