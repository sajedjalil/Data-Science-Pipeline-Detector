# Top 10 using forward selection with 4-fold cv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

print('Load data...')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

ntrain = train.shape[0]
target = train['TARGET']
ID_test = test['ID']

train = train.drop(['ID','TARGET'], axis=1)
test = test.drop('ID', axis=1)

# New features
print('Computing new features...')
train_test = pd.concat((train, test), axis=0)
features = train_test.columns
train_test['std'] = train_test.apply(lambda x: np.std(x), axis=1)

# Selected features (using forward selection with 4-Fold cv)
selected_features = ['var15',                 #0.70             250 rounds
                     'saldo_var30',           #0.812096189048   250 rounds
                     'std',                   #0.816972944498   250 rounds
                     'num_var22_ult3',        #0.829681738232   325 rounds
                     'imp_op_var39_ult1',     #0.833324036977   325 rounds
                     'num_var45_hace3',       #0.8347158495     325 rounds
                     'saldo_medio_var5_hace2',#0.836754399288   325 rounds
                     'var3',                  #0.838416590074   325 rounds
                     'saldo_medio_var8_ult3',
                     'ind_var41_0'            #0.836755316781 - 0.839971060754   325 rounds
]

train_test = train_test[selected_features]
train = train_test.iloc[:ntrain, :]
test = train_test.iloc[ntrain:, :]

# xgb parameters
params = {}
params['objective'] = "binary:logistic"
params['eta'] = 0.025
params['eval_metric'] = 'auc'
params['max_depth'] = 5
params['subsample'] = 0.8
params['colsample_bytree'] = 0.6
params['silent'] = 1

X_train = np.array(train)
X_test = np.array(test)
y_train = target.values

# Making predictions for 3 seeds (you can change this parameter below)
n_seed = 3
y_pred = np.zeros(test.shape[0])
for i in range(0,n_seed):
    print('Making prediction for seed',1+i,'...')
    # train machine learning
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(test)
    
    params['seed'] = 1+i
    num_rounds = 325
    watchlist = [(xg_train, 'train')]
    xgclassifier = xgb.train(params, xg_train, num_rounds, watchlist, verbose_eval=25);
    print('Predict...')
    new_pred = xgclassifier.predict(xg_test)
    y_pred += new_pred

# Making final prediction and writing the file
y_pred = y_pred/n_seed
pd.DataFrame({"ID": ID_test, "TARGET": y_pred}).to_csv('script1.csv',index=False)