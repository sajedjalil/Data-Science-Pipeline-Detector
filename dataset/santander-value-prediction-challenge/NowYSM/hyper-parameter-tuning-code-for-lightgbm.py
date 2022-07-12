# Reference Code : https://www.kaggle.com/seiya1998/lgbm-with-random-projection-and-aggregate-lb-1-41
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import random_projection

import lightgbm as lgb

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)

ntrain = len(X_train)
ntest = len(X_test)

print("Preparetion")
colsToRemove = []
for col in X_train.columns:
    if X_train[col].std() == 0: 
        colsToRemove.append(col)
        
# remove constant columns in the training set
X_train.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
X_test.drop(colsToRemove, axis=1, inplace=True) 

print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))


colsToRemove = []
colsScaned = []
dupList = {}

columns = X_train.columns

for i in range(len(columns)-1):
    v = X_train[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, X_train[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
                
# remove duplicate columns in the training set
X_train.drop(colsToRemove, axis=1, inplace=True) 

# remove duplicate columns in the testing set
X_test.drop(colsToRemove, axis=1, inplace=True)

print("Removed `{}` Duplicate Columns\n".format(len(dupList)))



print("Aggregate")
weight = ((X_train != 0).sum()/len(X_train)).values

tmp_train = X_train[X_train!=0]
tmp_test = X_test[X_test!=0]
tmp = pd.concat([X_train,X_test])#RandomProjection

X_train["weight_count"] = (tmp_train*weight).sum(axis=1)
X_test["weight_count"] = (tmp_test*weight).sum(axis=1)

X_train["count_not0"] = (X_train != 0).sum(axis=1)
X_test["count_not0"] = (X_test != 0).sum(axis=1)

X_train["sum"] = X_train.sum(axis=1)
X_test["sum"] = X_test.sum(axis=1)

X_train["var"] = tmp_train.var(axis=1)
X_test["var"] = tmp_test.var(axis=1)

X_train["mean"] = tmp_train.mean(axis=1)
X_test["mean"] = tmp_test.mean(axis=1)

X_train["std"] = tmp_train.std(axis=1)
X_test["std"] = tmp_test.std(axis=1)

X_train["max"] = tmp_train.max(axis=1)
X_test["max"] = tmp_test.max(axis=1)

X_train["min"] = tmp_train.min(axis=1)
X_test["min"] = tmp_test.min(axis=1)

del(tmp_train)
del(tmp_test)


print("Random_Projection")


n_com = 96

transformer = random_projection.GaussianRandomProjection(n_components = n_com)

RP = transformer.fit_transform(tmp)
rp = pd.DataFrame(RP)

columns = ["RandomProjection{}".format(i) for i in range(n_com)]
rp.columns = columns

rp_train = rp[:ntrain]
rp_test = rp[ntrain:]
rp_test.index = X_test.index

#concat RandomProjection and raw data
X_train = pd.concat([X_train,rp_train],axis=1)
X_test = pd.concat([X_test,rp_test],axis=1)

del(rp_train)
del(rp_test)


print("Modeling")
dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 64,
        "learning_rate" : 0.004,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : 1,
        "seed": 42,
         }
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result,
                      )
    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result

pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
print("LightGBM Training Completed...")


sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = pred_test
print(sub.head())
sub.to_csv('lightgbm1.34957.csv', index=False)




# {'boosting_type': 'goss',
    #       'max_depth' : -1,
    #       'objective': 'regression',
    #       'nthread': 3, # Updated from nthread
    #       'num_leaves': 64,
    #       'learning_rate': 0.05,
    #       'max_bin': 512,
    #       'subsample_for_bin': 200,
    #       'subsample': 1,
    #       'subsample_freq': 1,
    #       'colsample_bytree': 0.8,
    #       'reg_alpha': 5,
    #       'reg_lambda': 10,
    #       'min_split_gain': 0.5,
    #       'min_child_weight': 1,
    #       'min_child_samples': 5,
    #       'scale_pos_weight': 1,
    #       'num_class' : 1,
    #       'metric' : 'rmse'}
    
