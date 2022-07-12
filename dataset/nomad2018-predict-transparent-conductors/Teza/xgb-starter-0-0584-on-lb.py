# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


x_columns = [i for i in train.columns if i not in list(['id','formation_energy_ev_natom','bandgap_energy_ev'])]
label1 = 'formation_energy_ev_natom'
label2 = 'bandgap_energy_ev'


X = train[x_columns]
y = train[[label1,label2]]


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=101)

y_train1_values = np.log1p(y_train['formation_energy_ev_natom'].values)
y_train2_values = np.log1p(y_train['bandgap_energy_ev'].values)
y_valid1_values = np.log1p(y_valid['formation_energy_ev_natom'].values)
y_valid2_values = np.log1p(y_valid['bandgap_energy_ev'].values)



def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.1
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'rmse'
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
    return pred_test_y, model


preds, model = runXGB(X_train, y_train1_values, X_valid, y_valid1_values)
preds2, model2 = runXGB(X_train, y_train2_values, X_valid, y_valid2_values)

test_data = test[x_columns]
d_test = xgb.DMatrix(test_data)

preds_test1 = model.predict(d_test)
preds_test2 = model2.predict(d_test)
preds_test1 = np.exp(preds_test1)-1
preds_test2 = np.exp(preds_test2)-1

xgb = pd.DataFrame()
xgb['id'] = test['id']
xgb['formation_energy_ev_natom'] = preds_test1
xgb['bandgap_energy_ev'] = preds_test2
xgb.to_csv("xgb_starter.csv", index=False)


# Any results you write to the current directory are saved as output.