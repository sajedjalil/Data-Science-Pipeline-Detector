import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline, model_selection
from sklearn.metrics import log_loss, make_scorer
from subprocess import check_output
from sklearn.feature_extraction import DictVectorizer as DV
import random
random.seed(16)

#Forket for like for like comparison before boosting
cat_cols = ['v3','v24','v30','v31','v47','v52','v56','v66','v71','v74','v75','v79','v107','v110','v112','v113','v125']

def impute_most_freq_value(df,colname):
    c = df[colname].value_counts()
    return c.index[0]

def load_data():
    train = pd.read_csv('../input/train.csv')
    test  = pd.read_csv('../input/test.csv')

    train.drop(['v22'], axis=1, inplace=True)
    train.drop(['v91'], axis=1, inplace=True)
    test.drop(['v22'], axis=1, inplace=True)
    test.drop(['v91'], axis=1, inplace=True)

    nas = {}
    for colname in cat_cols:
        nas[colname] = impute_most_freq_value(train,colname)

    for colname in cat_cols:
        train[colname].fillna(nas[colname],inplace=True)

    for colname in cat_cols:
        test[colname].fillna(nas[colname],inplace=True)

    cat_train = train[cat_cols]
    cat_test = test[cat_cols]

    #put the numerical as matrix
    train.drop(cat_cols, axis=1, inplace=True)
    test.drop(cat_cols, axis=1, inplace=True)

    print(cat_train.describe())

    # transform the categorical to dict
    dict_train_data = cat_train.T.to_dict().values()
    dict_test_data = cat_test.T.to_dict().values()

    #vectorize
    vectorizer = DV(sparse = False)
    features = vectorizer.fit_transform(dict_train_data)
    vec_data = pd.DataFrame(features)
    vec_data.columns = vectorizer.get_feature_names()
    #vec_data.rename(columns={'changed': 'vec_changed'}, inplace=True)
    #vec_data.rename(columns={'id': 'vec_id'}, inplace=True)
    vec_data.index = train.index
    train = train.join(vec_data)

    features = vectorizer.transform(dict_test_data)
    vec_data = pd.DataFrame(features)
    vec_data.columns = vectorizer.get_feature_names()
    vec_data.index = test.index
    test = test.join(vec_data)

    #merge numerical and categorical sets
    trainend = int(0.75*len(train))
    valid_inds = list(train[trainend:].index.values)
    train_inds = list(train.loc[~train.index.isin(valid_inds)].index.values)

    train.fillna(-100,inplace=True)
    test.fillna(-100,inplace=True)

    return train, test, train_inds, valid_inds

train, test, train_inds, valid_inds = load_data()

valid = train.iloc[valid_inds]
train = train.iloc[train_inds]

trainlabels = train['target'].values
validlabels = valid['target'].values

X_train   = train.drop(['ID','target'],axis=1).values
X_valid   = valid.drop(['ID','target'],axis=1).values

ids       = test['ID'].values
X_test    = test.drop(['ID'],axis=1).values

id_test = ids[:]
y_train = train['target'].values

def flog_loss(ground_truth, predictions):
    flog_loss_ = log_loss(ground_truth, predictions) #, eps=1e-15, normalize=True, sample_weight=None)
    return flog_loss_

LL  = make_scorer(flog_loss, greater_is_better=False)

print("--- Features Set: %s minutes ---" % ((time.time() - start_time)/60))

rfr = RandomForestRegressor(n_estimators = 25, max_depth = 11, n_jobs = -1)
clf = pipeline.Pipeline([('rfr', rfr)])
param_grid = {}  
#param_grid = {'rfr__n_estimators' : [25], 'rfr__max_depth': [10]} #list(range(11,12,1))}
model = model_selection.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=LL)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)
print(len(y_pred))
min_y_pred = min(y_pred)
max_y_pred = max(y_pred)
min_y_train = min(y_train)
max_y_train = max(y_train)
print(min_y_pred, max_y_pred, min_y_train, max_y_train)
for i in range(len(y_pred)):
    y_pred[i] = min_y_train + (((y_pred[i] - min_y_pred)/(max_y_pred - min_y_pred))*(max_y_train - min_y_train))
pd.DataFrame({"ID": id_test, "PredictedProb": y_pred}).to_csv('submission.csv',index=False)
print("--- Training & Testing: %s minutes ---" % ((time.time() - start_time)/60))

feature_names = np.array(train.drop(['ID','target'],axis=1).columns.values.tolist())
importances = model.best_estimator_.named_steps['rfr'].feature_importances_
important_names = feature_names[importances > (np.mean(importances)/2)]
drop_names = [a for a in feature_names if a not in important_names]
print(importances)
print("------------------------------------")
print(drop_names)
    