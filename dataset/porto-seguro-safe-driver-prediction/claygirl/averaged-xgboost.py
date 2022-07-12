import numpy as np 
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit

def ginic(actual, pred):
    actual = np.asarray(actual) 
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalized(a, p):
    if p.ndim == 2:
        p = p[:,1] 
    return ginic(a, p) / ginic(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return "gini", gini_score

def preprocess_input(trainset, testset):
    for cat in categorical:
        trainset = pd.concat([trainset, pd.get_dummies(trainset[cat], prefix=cat)], axis=1)
        testset = pd.concat([testset, pd.get_dummies(testset[cat], prefix=cat)], axis=1)
        cond_prob = trainset.groupby(cat)["target"].mean().reset_index()
        cond_prob = cond_prob.rename(columns={"target": cat + "_prob"})
        trainset = pd.merge(trainset, cond_prob, on=cat, how="left")
        testset = pd.merge(testset, cond_prob, on=cat, how="left")
    return trainset, testset
    
# ps_car_08_cat - has only two values
categorical = ["ps_car_02_cat",
               "ps_car_09_cat",
               "ps_ind_04_cat",
               "ps_ind_05_cat",
               "ps_car_03_cat",
               "ps_car_05_cat",
               "ps_car_07_cat",
               "ps_car_11_cat",
               "ps_car_10_cat",
               "ps_car_04_cat",
               "ps_car_01_cat",
               "ps_ind_02_cat",
               "ps_car_06_cat"]
               
raw_train = pd.read_csv("../input/train.csv")
raw_test = pd.read_csv("../input/test.csv")
train, test = preprocess_input(raw_train, raw_test)

y = train.target.values
id_test = test["id"].values

train = train.drop(["id","target"], axis=1)
test = test.drop(["id"], axis=1)
X = train.values
d_test = xgb.DMatrix(test.values)

params = {"max_depth": 6, 
          "eta": 0.1, 
          "objective": "binary:logistic",
          "silent": False,
          "subsample": 0.9,
          "colsample_bytree": 0.9,
          "colsample_bylevel": 0.9
         }

sub = pd.DataFrame()
sub["id"] = id_test
sub["target"] = np.zeros_like(id_test)

kfold = 5
sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.2, random_state=0)
for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    print("[Fold %d/%d]" % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    watchlist = [(d_train, "train"), (d_valid, "valid")]

    model = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=20, feval=gini_xgb, maximize=True)

    p_test = model.predict(d_test)
    sub["target"] += p_test/kfold

sub.to_csv("submission.csv", index=False)