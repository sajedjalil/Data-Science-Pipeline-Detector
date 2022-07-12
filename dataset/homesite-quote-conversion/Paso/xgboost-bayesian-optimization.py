# XGBoost Bayesian Optimization
# All credits go to https://github.com/mpearmain/BayesBoost for BayesBoost (Also download bayes_opt from there)
# and to the Autor of https://www.kaggle.com/mpearmain/homesite-quote-conversion/xgboost-benchmark
# which in both cases is mpearmain

# The runtime of this is way to long for kaggle scripts and bayes_opt isn't available. Please run it locally.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from bayes_opt import BayesianOptimization

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop('QuoteNumber', axis=1)

# Lets play with some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek


test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

train = train.fillna(0)
test = test.fillna(0)

for f in train.columns:
    if train[f].dtype == 'object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

print("Scaling")
pca = StandardScaler()
features = list(train.columns.values)
train = pca.fit_transform(train, y)
test = pca.transform(test)


def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree,
              silent=True,
              nthread=-1):
    return cross_val_score(xgb.XGBClassifier(max_depth=int(max_depth),
                                             learning_rate=learning_rate,
                                             n_estimators=int(n_estimators),
                                             silent=silent,
                                             nthread=nthread,
                                             gamma=gamma,
                                             min_child_weight=min_child_weight,
                                             max_delta_step=max_delta_step,
                                             subsample=subsample,
                                             colsample_bytree=colsample_bytree),
                           train,
                           y,
                           "roc_auc",
                           cv=5).mean()


# Load data set and target values

xgboostBO = BayesianOptimization(xgboostcv,
                                 {'max_depth': (5, 10),
                                  'learning_rate': (0.01, 0.3),
                                  'n_estimators': (50, 1000),
                                  'gamma': (1., 0.01),
                                  'min_child_weight': (2, 10),
                                  'max_delta_step': (0, 0.1),
                                  'subsample': (0.7, 0.8),
                                  'colsample_bytree' :(0.5, 0.99)
                                  })

xgboostBO.maximize()
print('-'*53)

print('Final Results')
print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])