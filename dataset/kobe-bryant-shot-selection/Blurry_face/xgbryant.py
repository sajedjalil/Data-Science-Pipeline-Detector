import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.cross_validation import cross_val_score, train_test_split
data = pd.read_csv('../input/data.csv')
total_actions = dict(data.action_type.value_counts())
# TODO: tune this parameter
threshold = 100
# TODO: check if it's needed
data['type'] = data.apply(lambda row: row['action_type'] if total_actions[row['action_type']] >= threshold \
                          else row['combined_shot_type'], axis=1)
data['time_remaining'] = data.apply(lambda row: row['minutes_remaining'] * 60 + row['seconds_remaining'], axis=1)

# TODO: tune this parameter
threshold = 3
# TODO: find out why he cant hit @ 14 secs to go
anomaly = 14
data['last_moment'] = data.apply(lambda row: row['time_remaining'] < threshold or row['time_remaining'] == anomaly, axis=1)
data['away'] = data.matchup.str.contains('@')
data['season'] = data.apply(lambda row: int(row['season'].split('-')[0]), axis=1)

# remove outliers
data['shot_distance'] = data.apply(lambda row: 28 if row['shot_distance'] > 28 else row['shot_distance'], axis=1)
def get_acc(data, field):
    ct = pd.crosstab(data.shot_made_flag, data[field]).apply(lambda x: x / x.sum(), axis=0)
    x, y = ct.columns, ct.values[1, :]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel(field)
    plt.ylabel('% shots made')
    plt.show()
    
def sort_encode(data, field):
    ct = pd.crosstab(data.shot_made_flag, data[field]).apply(lambda x: x / x.sum(), axis=0)
    temp = list(zip(ct.values[1, :], ct.columns))
    temp.sort()
    new_map = {}
    for index, (acc, old_number) in enumerate(temp):
        new_map[old_number] = index
    new_field = field + '_sort_enumerated'
    data[new_field] = data[field].map(new_map)
    get_acc(data, new_field)

data.drop([
        'action_type',
        'combined_shot_type',
        'game_event_id',
        'game_id',
        'lat',
        'lon',
        'minutes_remaining',
        'seconds_remaining',
        'time_remaining',
        'team_id',
        'team_name',
        'matchup',
        'game_date',
        'shot_type',
        'playoffs',
        'season',
        # TODO: find out whether these two features matter or not
        'loc_x',
        'loc_y',
    ], axis=1, inplace=True)

dummies = [
    'period',
    'type',
    'shot_zone_area',
    'shot_zone_basic',
    'shot_zone_range',
    'opponent', #TODO: check if it's needed
]
dummie_counter = {}
for dummy in dummies:
    dummie_counter[dummy] = len(data[dummy].unique())
data = pd.get_dummies(data, columns=dummies)
data.head()
for col in data.columns.tolist():
    if col not in ['shot_made_flag', 'shot_id']:
        print(col + ': ' + str(len(data[col].unique())))
# split into train and test
train = data[~data['shot_made_flag'].isnull()]
test = data[data['shot_made_flag'].isnull()]
print('train size: ' + str(len(train)))
print('test size:  ' + str(len(test)))

# prepare data for estimators
target = 'shot_made_flag'
features = data.columns.tolist()
features.remove(target)
features.remove('shot_id')
X_test = test[features]
X_train = train[features]
y_train = train[[target]]['shot_made_flag'].values
print(X_train.shape)
print(y_train.shape)
def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds, metrics=['logloss'])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    # Test params
    X_train, X_valid, y_train, y_valid = train_test_split(dtrain[predictors], dtrain[target], test_size=0.2)
    alg.fit(X_train, y_train, eval_metric='logloss')
    y_pred = alg.predict_proba(X_valid)[:,1]
    
    result = log_loss(y_valid, y_pred)
    print(result)
    return result
# fit, predict and generate submission file
params = {
    'base_score': 0.5, 
    'colsample_bylevel': 1,
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_child_weight': 1,
    'n_estimators': 400,
    'nthread': -1,
    'objective': 'binary:logistic',
    'seed': 27,
    'silent': True,
    'subsample': 0.8
}
clf = xgb.XGBClassifier()
clf.set_params(**params)
clf.fit(X_train, y_train)

test.shot_made_flag = [i[1] for i in clf.predict_proba(X_test)]

test[['shot_id', 'shot_made_flag']].to_csv('sub.csv', index=False)
predictions_train = clf.predict_proba(X_train)
features_train = pd.DataFrame({'shot_made_flag': predictions_train[:, 1]})
features_train[['shot_made_flag']].to_csv('features_train.csv', index=False)
