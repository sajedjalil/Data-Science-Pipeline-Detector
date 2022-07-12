import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = np.log(train['loss'])
indices = test['id']

train.drop(['id', 'loss'], axis = 1, inplace = True)
test.drop('id', axis = 1, inplace = True)

categorical = [col for col in train.columns if 'cat' in col]
continuous = [col for col in train.columns if 'cont' in col]

num_train_rows = train.shape[0]
df = pd.concat((train, test)).reset_index(drop=True)

mms = MinMaxScaler()
for col in continuous:
    shift = np.abs(np.floor(min(df[col])))
    df[col] = mms.fit_transform(
        (stats.boxcox(df[col] + shift + 1)[0]).reshape(-1, 1))
    
le = LabelEncoder()
for col in categorical:
    df[col] = le.fit_transform(df[col])

x_train = np.array(df.iloc[:num_train_rows,:])
x_test = np.array(df.iloc[num_train_rows:,:])

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.075
params['gamma'] = 0.5290
params['min_child_weight'] = 4.2922
params['num_parallel_tree'] = 1
params['colsample_bytree'] = 0.3085
params['subsample'] = 0.9930
params['max_depth'] = 7
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 1337

dtrain = xgb.DMatrix(x_train, label = y_train)
dtest = xgb.DMatrix(x_test)

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

res = xgb.cv(params, 
            dtrain, 
            num_boost_round = 100, 
            nfold = 4, 
            seed = 1337, 
            stratified = False, 
            early_stopping_rounds = 15, 
            verbose_eval = 10, 
            show_stdv = True, 
            feval = xg_eval_mae, 
            maximize = False)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]
print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

model = xgb.train(params, dtrain, best_nrounds)

predictions = np.exp(model.predict(dtest))

submission = pd.DataFrame({"id":indices, "loss":predictions})
submission.to_csv('submission.csv', index = None)

plt.xlim(-1000, 15000)
sns.kdeplot(np.exp(y_train))
sns.kdeplot(predictions)
plt.savefig("submission.png")

