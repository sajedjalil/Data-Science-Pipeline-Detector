import numpy as np
import pandas as pd
import xgboost as xgb

traincsv = "../input/train.csv"
testcsv = "../input/test.csv"
#traincsv = "train.csv"
#testcsv = "test.csv"

df_train = pd.read_csv(traincsv)
df_test = pd.read_csv(testcsv)

ignored_columns = ['ID', 'TARGET']
C = df_train.columns
# remove constant columns
eps = 1e-6
dropped_columns = set()
print('Identifing low-variance columns...', end=' ')
for c in C:
    if df_train[c].var() < eps:
        # print('.. %-30s: too low variance ... column ignored'%(c))
        dropped_columns.add(c)
print('done!')
C = list(set(C) - dropped_columns - set(ignored_columns))
# remove duplicate columns
print('Identifying duplicate columns...', end=' ')
for i, c1 in enumerate(C):
    f1 = df_train[c1].values
    for j, c2 in enumerate(C[i+1:]):
        f2 = df_train[c2].values
        if np.all(f1 == f2):
            dropped_columns.add(c2)
print('done!')
C = list(set(C) - dropped_columns - set(ignored_columns))
print('# columns dropped: %d'%(len(dropped_columns)))
print('# columns retained: %d'%(len(C)))
df_train.drop(dropped_columns, axis=1, inplace=True)
df_test.drop(dropped_columns, axis=1, inplace=True)

# Splitting data columns
y_train = df_train['TARGET'].values
x_train = df_train.drop(['ID','TARGET'], axis=1).values
id_test = df_test['ID']
x_test = df_test.drop(['ID'], axis=1).values

# zero-count
zero_train = (x_train == 0).astype(int).sum(axis=1)
zero_train = pd.DataFrame(zero_train, columns=["ZERO"])
x_train = pd.DataFrame(x_train)
x_train = pd.concat([x_train, zero_train], axis=1)

zero_test = (x_test == 0).astype(int).sum(axis=1)
zero_test = pd.DataFrame(zero_test, columns=["ZERO"])
x_test = pd.DataFrame(x_test)
x_test = pd.concat([x_test, zero_test], axis=1)

d_train = xgb.DMatrix(x_train, label=y_train)
watchlist  = [(d_train,'train')]

params = {}
params['objective'] = 'binary:logistic'
params['booster'] = 'gbtree'
params['eval_metric'] = 'auc'
params['eta'] = 0.0201
params['max_depth'] = 5
params['subsample'] = 0.6815
params['colsample_bytree'] = 0.7
params['verbose'] = 2
#params['show_progress'] = True
#params['print_every_n'] = 1
params['maximise'] = False

clf = xgb.train(params, d_train, 572, watchlist)

d_test = xgb.DMatrix(x_test)
y_pred = clf.predict(d_test)
submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)
