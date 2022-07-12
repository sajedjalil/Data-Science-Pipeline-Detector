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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn import model_selection, preprocessing


pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = 999

x_train = pd.read_csv("../input/train.csv")
x_test = pd.read_csv("../input/test.csv")

x_all = pd.concat([x_train, x_test], ignore_index = True)

def get_integer_features(dataset):
    for c in dataset.columns:
        if dataset[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder() 
            lbl.fit(list(dataset[c].values)) 
            dataset[c] = lbl.transform(list(dataset[c].values))
    return dataset
        #x_train.drop(c,axis=1,inplace=True)
        
x_all = get_integer_features(x_all)    

x_train = x_all[:len(x_train)]
x_test = x_all[-len(x_test):]

        
x_train.dtypes.reset_index()

y_train = x_train["y"]

id_test = x_test["ID"]
x_train = x_train.drop(["y", "ID"], axis=1)
x_test = x_test.drop(["ID", "y"], axis=1)


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=5)


model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=600)

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

y_predict = model.predict(dtest)

output = pd.DataFrame({'id': id_test, 'y': y_predict})

output.to_csv('submission.csv', index=False)