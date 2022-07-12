#! /home/xuhaotian/anaconda2/bin/env python
import pandas as pd
import xgboost as xgb
import operator
from matplotlib import pylab as plt
import numpy as np
from sklearn import ensemble, preprocessing

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

def get_data():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv('../input/test.csv')

    features = list(train.columns[2:])
    test_features = list(test.columns[1:])	
    y_train = train.Hazard
    
    # statistic transformation for catagorical variable of training dataset
    for feat in train.select_dtypes(include=['object']).columns:
        m = train.groupby([feat])['Hazard'].mean()
        train[feat].replace(m,inplace=True)
        test[feat].replace(m,inplace=True)


    x_train = train[features]
    x_test = test[test_features]
    idx = test.Id
    return  features, x_train, y_train, x_test, idx


features, x_train, y_train, x_test, idx = get_data()
ceate_feature_map(features)
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.01
params["min_child_weight"] = 5
params["subsample"] = 0.8
#params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 1
params["silent"] = 1
params["max_depth"] = 7
plst = list(params.items())
offset = 5000
num_rounds = 2000
#xgb_params = {"objective": "reg:linear", "eta": 0.01, "max_depth": 8, "seed": 42, "silent": 1}
#num_rounds = 1000
#xgb_params = {"objective": "reg:linear", "eta": 0.005, "max_depth": 9, "seed": 42, "silent": 1}
#dtrain = xgb.DMatrix(x_train, label=y_train)

train = np.array(x_train)
test = np.array(x_test)
train = train.astype(float)
test = test.astype(float)

xgtrain = xgb.DMatrix(train[offset:,:], label=y_train[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=y_train[:offset])
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
#gbdt = xgb.train(xgb_params, dtrain, num_rounds)
gbdt = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)

#importance = gbdt.get_fscore(fmap='xgb.fmap')
#importance = sorted(importance.items(), key=operator.itemgetter(1))

#df = pd.DataFrame(importance, columns=['feature', 'fscore'])
#df['fscore'] = df['fscore'] / df['fscore'].sum()

xgtest = xgb.DMatrix(test)
#debug = pd.DataFrame({"T1_V4": x_test.T1_V4,"Id": idx})
#debug.to_csv('debug.csv', index=False)
#preds = np.expm1(gbdt.predict(xgtest, ntree_limit=gbdt.best_iteration))
preds = gbdt.predict(xgtest)
preds = pd.DataFrame({"Id": idx, "Hazard": preds})
preds.to_csv('prediction2.csv', index=False)

#plt.figure()
#df.plot()
#df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
#plt.title('XGBoost Feature Importance')
#plt.xlabel('relative importance')
#plt.gcf().savefig('feature_importance_xgb2.png')
