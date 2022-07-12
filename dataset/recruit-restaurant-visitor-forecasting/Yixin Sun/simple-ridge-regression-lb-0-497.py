import numpy as np
import pandas as pd
from sklearn import utils, preprocessing, linear_model, neighbors, metrics, model_selection

######################################################################
###                            Read Files                          ###
######################################################################
np.random.seed(2018)

data = {
    'air_visit': pd.read_csv('../input/air_visit_data.csv', parse_dates=['visit_date']),
    'holidays': pd.read_csv('../input/date_info.csv', parse_dates=['calendar_date']).rename(
        columns={'calendar_date': 'visit_date'}),
    'submission': pd.read_csv('../input/sample_submission.csv')
}

data['holidays'].drop(['day_of_week'], axis=1, inplace=True)

######################################################################
###                        Data Preparation                        ###
### Contributions from:                                            ###
### the1owl - Surprise Me                                          ###
### https://www.kaggle.com/the1owl/surprise-me                     ###
######################################################################
# Add day of week, month into training set and test set
data['submission']['visit_date'] = data['submission']['id'].apply(lambda x:x[-10:])
data['submission']['visit_date'] = pd.to_datetime(data['submission']['visit_date'])
data['submission']['air_store_id'] = data['submission']['id'].apply(lambda x:x[:-11])
data['submission'].drop(['id'], axis=1, inplace=True)
for df in ['air_visit', 'submission']:
    data[df]['day_of_week'] = data[df]['visit_date'].dt.dayofweek
    data[df]['month'] = data[df]['visit_date'].dt.month
    data[df]['year'] = data[df]['visit_date'].dt.year
    
# Aggregate min, max, median, and mean of visitors
unique_stores = data['submission']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'day_of_week': [i] * len(unique_stores)}) \
                    for i in range(7)], ignore_index=True)

funcs = {
    'min': 'visitors_min',
    'max': 'visitors_max',
    'mean': 'visitors_mean',
    'median': 'visitors_median'
}
for func in funcs:
    tmp = data['air_visit'].groupby(['air_store_id', 'day_of_week'], as_index=False).agg(
    {'visitors': func}).rename(columns={'visitors': funcs[func]})
    stores = stores.merge(tmp, how='left', on=['air_store_id', 'day_of_week'])
    
# Merge training and test sets with holidays
train = pd.merge(data['air_visit'], data['holidays'], how='left', on='visit_date')
test = pd.merge(data['submission'], data['holidays'], how='left', on='visit_date')

# Merge training and test sets with store information
train = pd.merge(train, stores, how='inner', on=['air_store_id', 'day_of_week'])
test = pd.merge(test, stores, how='inner', on=['air_store_id', 'day_of_week'])

'''
# Add interaction terms
for df in [train, test]:
    df['min_min'] = df['visitors_min'] * df['visitors_min']
    df['mean_mean'] = df['visitors_mean'] * df['visitors_mean']
    df['median_median'] = df['visitors_median'] * df['visitors_median']
    df['min_max'] = df['visitors_min'] * df['visitors_max']
    df['min_mean'] = df['visitors_min'] * df['visitors_mean']
    df['min_median'] = df['visitors_min'] * df['visitors_median']
    df['max_mean'] = df['visitors_max'] * df['visitors_mean']
    df['max_median'] = df['visitors_max'] * df['visitors_median']
    df['mean_median'] = df['visitors_mean'] * df['visitors_median']
'''

# Seperate data according to air_store_id.
# X_train, X_test: <(float)air_store_id, data for this id>
# y_train: <(float)air_store_id, (DataFrame)targets>
print('Start seperating data according to store id.')
X_train, X_test, y_train = {}, {}, {}
le = preprocessing.LabelEncoder()
drop_columns = ['air_store_id', 'visit_date', 'visitors']
categorical_columns = ['month', 'day_of_week']

for store_id in train['air_store_id'].unique():
    if store_id in X_train:
        continue
    tmp1 = train[train['air_store_id'] == store_id]
    tmp1 = utils.shuffle(tmp1).reset_index(drop=True)
    tmp2 = test[test['air_store_id'] == store_id]
    y_train[store_id] = np.log1p(tmp1['visitors'])
    
    tmp = pd.concat([tmp1, tmp2], ignore_index=True)
    tmp = pd.get_dummies(tmp, columns=categorical_columns)
    tmp['year'] = le.fit_transform(tmp['year'])
    tmp.drop(drop_columns, axis=1, inplace=True)
    
    X_train[store_id] = tmp[:tmp1.shape[0]]
    X_test[store_id] = tmp[tmp1.shape[0]:]
print('Seperation done!')
    
# Impute missing data in X_test
print('Start imputing missing data in test set.')
missing_columns = test.columns[test.isnull().any()]
missing_ids = test['air_store_id'][test[missing_columns[0]].isnull()].unique()
for store_id in missing_ids:
    known = X_train[store_id].drop(missing_columns, axis=1)
    unknown = X_test[store_id][X_test[store_id][missing_columns[0]].isnull()].drop(missing_columns, axis=1)
    neigh = neighbors.NearestNeighbors(n_neighbors=10, algorithm='brute', n_jobs=-1, metric='euclidean')
    neigh.fit(known)
    for idx in unknown.index:
        idx_nei = neigh.kneighbors(unknown.loc[idx].values.reshape(1, -1), return_distance=False)
        X_test[store_id].loc[idx] = X_test[store_id].loc[idx].fillna(
            X_train[store_id].iloc[idx_nei[0]][missing_columns].mean())
print('Imputation done!')
            
######################################################################
###                             Training                           ###
### Train a ridge regression model for each store  id individually ### 
######################################################################
# Define evaluation metric
def RMSE(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5
rmse = metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False)

# Training
alphas = np.logspace(-2, 7, base=2, num=60)    # you may want to modify this
ridge_cv = linear_model.RidgeCV(alphas=alphas, scoring=rmse, cv=3)

print('Start training models')
model = {}; iteration = 0
for store_id in train['air_store_id'].unique():
    # select the optimal ridge regression model using cv
    ridge_cv.fit(X_train[store_id], y_train[store_id])
    model_ridge = linear_model.Ridge(alpha=ridge_cv.alpha_, max_iter=0x7fffffff)
    model_ridge.fit(X_train[store_id], y_train[store_id])
    model[store_id] = model_ridge
    
    print('Best l2 term for {}: {}'.format(store_id, ridge_cv.alpha_))
    print('Iteration {} is finished'.format(iteration))
    iteration += 1

# Estimate model performance on entire training set
y_true, y_pred = pd.DataFrame(), pd.DataFrame()
for store_id in train['air_store_id'].unique():
    y_true = pd.concat([y_true, y_train[store_id]])
    yhat = model[store_id].predict(X_train[store_id])
    y_pred = pd.concat([y_pred, pd.DataFrame(yhat)])
print('RMSLE on entire training set:', RMSE(y_true, y_pred))



# Make predictions on test set and generate submission
test['visitors'] = 0.0
for store_id in train['air_store_id'].unique():
    yhat = model[store_id].predict(X_test[store_id])
    test['visitors'][test['air_store_id'] == store_id] = yhat

test['visit_date'] = test['visit_date'].astype(str)
test['id'] = test['air_store_id'] + '_' + test['visit_date']
test_sub = test.drop([col for col in test.columns if col not in ['id', 'visitors']], axis=1)
test_sub['visitors'] = np.expm1(test_sub['visitors']).clip(lower=0.)

submission = pd.read_csv('../input/sample_submission.csv').drop('visitors', axis=1)
submission = submission.merge(test_sub, on='id', how='inner')
submission.to_csv('submission.csv', index=False)