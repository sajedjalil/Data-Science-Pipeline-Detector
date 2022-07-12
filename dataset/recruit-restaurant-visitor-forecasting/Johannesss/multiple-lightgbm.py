import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn import model_selection, ensemble, neighbors
from sklearn.linear_model import LogisticRegression

from subprocess import check_output
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import preprocessing

data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }
data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])
data['hr'].drop('hpg_store_id',  axis=1, inplace=True)
data['ar'] = data['ar'].append(data['hr'])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
test_id = data['tes']['id']
data['tes'].drop('id', axis=1, inplace=True)
print ('Data loaded - number visits: ' + str(data['tra'].shape[0]))

# Create single data set with all relevant base data:
data['tra']['visit_datetime'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow']     = data['tra']['visit_datetime'].dt.dayofweek
data['ar']['res_visit_datetime'] = pd.to_datetime(data['ar']['visit_datetime'])
data['ar']['reserve_datetime']   = pd.to_datetime(data['ar']['reserve_datetime'])
data['ar']['visit_date']         = data['ar']['res_visit_datetime'].dt.date
data['ar']['reserve_diff'] = data['ar'].apply(lambda r: (r['res_visit_datetime']
                                                         - r['reserve_datetime']).days, 
                                        axis=1)
data['ar'].drop('visit_datetime',  axis=1, inplace=True)
data['ar'].drop('reserve_datetime',  axis=1, inplace=True)
data['ar'].drop('res_visit_datetime',  axis=1, inplace=True)
avg_reserv = data['ar'].groupby(['air_store_id','visit_date'], 
                                as_index=False).mean().reset_index()
data['ar'] = data['ar'].groupby(['air_store_id','visit_date'], 
                                as_index=False).sum().reset_index()
data['ar'] = data['ar'].drop(['reserve_diff'],axis=1)
data['ar'] = data['ar'].drop(['index'],axis=1)
data['ar']['reserve_diff'] = avg_reserv['reserve_diff']  
data['ar']['visit_date'] = data['ar']['visit_date'].astype(str)    

data['tes']['visit_datetime'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow']     = data['tes']['visit_datetime'].dt.dayofweek

prep_df = pd.merge(data['tra'], data['ar'],  how='left', on=['air_store_id', 'visit_date'])
prep_df = pd.merge(prep_df,     data['as'],  how='inner', on='air_store_id')
prep_df = pd.merge(prep_df,     data['hol'], how='left',  on='visit_date')
prep_df = prep_df[prep_df['visit_date'] >= '2016-06-29']
print ('Data merged - number visits in train: ' + str(prep_df.shape[0]))
predict_data = pd.merge(data['tes'],  data['ar'],   how='left', on=['air_store_id', 'visit_date'])
predict_data = pd.merge(predict_data, data['as'],   how='inner', on='air_store_id')
predict_data = pd.merge(predict_data, data['hol'],  how='left', on='visit_date')
print ('Data merged - number visits in test: ' + str(predict_data.shape[0]))

tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].min().rename(
    columns={'visitors': 'min_visitors'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id', 'dow'])
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].mean().rename(
    columns={'visitors': 'mean_visitors'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id', 'dow'])
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].median().rename(
    columns={'visitors': 'median_visitors'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id', 'dow'])
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].max().rename(
    columns={'visitors': 'max_visitors'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id', 'dow'])
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].count().rename(
    columns={'visitors': 'count_observations'})
prep_df = pd.merge(prep_df, tmp, how='left', on=['air_store_id', 'dow'])
predict_data = pd.merge(predict_data, tmp, how='left', on=['air_store_id', 'dow'])

prep_df.drop('dow',  axis=1, inplace=True)
predict_data.drop('dow',  axis=1, inplace=True)

# Encode fields:
prep_df['month'] = prep_df['visit_datetime'].dt.month
prep_df['day']   = prep_df['visit_datetime'].dt.day
#prep_df['woy']   = prep_df['visit_datetime'].dt.weekofyear
prep_df.drop('visit_datetime',      axis=1, inplace=True)   
predict_data['month'] = predict_data['visit_datetime'].dt.month
predict_data['day']   = predict_data['visit_datetime'].dt.day
#predict_data['woy']   = predict_data['visit_datetime'].dt.weekofyear
predict_data.drop('visit_datetime', axis=1, inplace=True)
prep_df.fillna(-1, inplace=True)
predict_data.fillna(-1, inplace=True)

# Encode labels of categorical columns:
cat_features = [col for col in ['air_genre_name', 'air_area_name', 'day_of_week']]
for column in cat_features:
    temp_prep = pd.get_dummies(pd.Series(prep_df[column]))
    prep_df = pd.concat([prep_df,temp_prep],axis=1)
    prep_df = prep_df.drop([column],axis=1)
    temp_predict = pd.get_dummies(pd.Series(predict_data[column]))
    predict_data = pd.concat([predict_data,temp_predict],axis=1)
    predict_data = predict_data.drop([column],axis=1)
    for missing_col in temp_prep:     # Make sure the columns of train and test are identical
        if missing_col not in predict_data.columns:
            predict_data[missing_col] = 0
    for missing_col in temp_predict:     # Make sure the columns of train and test are identical
        if missing_col not in prep_df.columns:
            prep_df[missing_col] = 0        

# Try runs without these columns:
#prep_df = prep_df.drop(['reserve_visitors'],axis=1)
#prep_df = prep_df.drop(['reserve_diff'],axis=1)
#prep_df = prep_df.drop(['latitude'],axis=1)
#prep_df = prep_df.drop(['longitude'],axis=1)
#predict_data = predict_data.drop(['reserve_visitors'],axis=1)
#predict_data = predict_data.drop(['reserve_diff'],axis=1)
#predict_data = predict_data.drop(['latitude'],axis=1)
#predict_data = predict_data.drop(['longitude'],axis=1)  

prep_df['visitors'] = np.log1p(prep_df['visitors'])

#early_prep = prep_df[prep_df['visit_date'] < '2016-06-29']
#prep_df[prep_df['visit_date'] < '2016-06-29']['visitors'] = prep_df[prep_df['visit_date'] < '2016-06-29']['visitors'] * 2
#prep_df = prep_df[prep_df['visit_date'] >= '2016-06-29']
#prep_df = prep_df.append(early_prep)
#prep_df.head()

prep_df.drop(['visit_date'], axis=1, inplace=True)
label_enc = preprocessing.LabelEncoder()
label_enc.fit(prep_df['air_store_id'])
prep_df['air_store_id'] = label_enc.transform(prep_df['air_store_id'])
prep_cols = prep_df.columns

predict_data.drop(['visit_date'], axis=1, inplace=True)  
predict_data['air_store_id'] = label_enc.transform(predict_data['air_store_id'])
 
X_train = prep_df.drop(['visitors'], axis=1)
y_train = prep_df['visitors'].values    
X_test = predict_data.drop(['visitors'], axis=1)

print('Data preparation done')
# Submissions are evaluated using RMSLE:
def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5
    
lgb_params1 = {}
lgb_params1['application'] = 'regression'
lgb_params1['boosting'] = 'gbdt'
lgb_params1['learning_rate'] = 0.015
lgb_params1['num_leaves'] = 32
lgb_params1['min_sum_hessian_in_leaf'] = 2e-2
lgb_params1['min_gain_to_split'] = 0
lgb_params1['bagging_fraction'] = 0.9
lgb_params1['feature_fraction'] = 0.9
lgb_params1['num_threads'] = 8
lgb_params1['metric'] = 'rmse'

lgb_params2 = {}
lgb_params2['application'] = 'regression'
lgb_params2['boosting'] = 'gbdt'
lgb_params2['learning_rate'] = 0.02
lgb_params2['lambda_l1'] = 0.5
lgb_params2['num_leaves'] = 32
lgb_params2['min_gain_to_split'] = 0
lgb_params2['bagging_fraction'] = 0.8
lgb_params2['feature_fraction'] = 0.8
lgb_params2['num_threads'] = 4
lgb_params2['metric'] = 'rmse'

lgb_params3 = {}
lgb_params3['application'] = 'regression'
lgb_params3['boosting'] = 'gbdt'
lgb_params3['learning_rate'] = 0.022
lgb_params3['num_leaves'] = 32
lgb_params2['lambda_l2'] = 0.3
lgb_params3['bagging_freq'] = 8
lgb_params3['min_gain_to_split'] = 0
lgb_params3['bagging_fraction'] = 0.8
lgb_params3['feature_fraction'] = 0.8
lgb_params3['num_threads'] = 4
lgb_params3['metric'] = 'rmse'

def do_train(X_train, X_valid, lgb_params, rounds):
    X_t = X_train.drop(['visitors'], axis=1)
    y_t = X_train['visitors'].values
    d_train = lgb.Dataset(X_t, y_t)
    X_v = X_valid.drop(['visitors'], axis=1)
    y_v = X_valid['visitors'].values
    d_valid = lgb.Dataset(X_v, y_v)
    watchlist = [d_train, d_valid]
    lgb_model = lgb.train(lgb_params, train_set=d_train, num_boost_round=rounds, 
                          valid_sets=watchlist, verbose_eval=1000, early_stopping_rounds = 300)
    test_pred = lgb_model.predict(X_v)
    rmsle = RMSLE(y_v, test_pred)
    print(X_t.columns)
    print(lgb_model.feature_importance())
    return rmsle, lgb_model

#print('Train with neighbors...')
#X_train, X_valid = train_test_split(prep_df, test_size=0.3, random_state=74, shuffle=True)
#model_gb = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)                                            
#X_t = X_train.drop(['visitors'], axis=1)
#y_t = X_train['visitors'].values                                            
#model_gb.fit(X_t, y_t)
#X_v = X_valid.drop(['visitors'], axis=1)
#y_v = X_valid['visitors'].values
#val_pred = model_gb.predict(X_v)
#rmsle = RMSLE(y_v, val_pred)
#test_pred = model_gb.predict(X_test)
#print('Test RMSLE: %.3f' % rmsle)

X_train, X_valid = train_test_split(prep_df, test_size=0.3, random_state=74, shuffle=True)
rmsle, lgb_model1 = do_train(X_train, X_valid, lgb_params1, 12000)
test_pred1 = np.expm1(lgb_model1.predict(X_test))
print('Test RMSLE: %.3f' % rmsle)
    
X_train, X_valid = train_test_split(prep_df, test_size=0.3, random_state=2121, shuffle=True)
rmsle, lgb_model2 = do_train(X_train, X_valid, lgb_params2, 10000)
test_pred2 = np.expm1(lgb_model2.predict(X_test))
print('Test RMSLE: %.3f' % rmsle)   

X_train, X_valid = train_test_split(prep_df, test_size=0.3, random_state=4, shuffle=True)
rmsle, lgb_model3 = do_train(X_train, X_valid, lgb_params3, 8000)
test_pred3 = np.expm1(lgb_model3.predict(X_test))
print('Test RMSLE: %.3f' % rmsle)   

X_train, X_valid = train_test_split(prep_df, test_size=0.3, random_state=19, shuffle=True)
rmsle, lgb_model4 = do_train(X_train, X_valid, lgb_params3, 8000)
test_pred4 = np.expm1(lgb_model4.predict(X_test))
print('Test RMSLE: %.3f' % rmsle)  

#test_pred = (test_pred3 + test_pred4) / 2
test_pred = (test_pred1 + test_pred2 + test_pred3 + test_pred4) / 4
result = pd.DataFrame({"id": test_id, "visitors": test_pred})   
result.to_csv('LGB_sub.csv', index=False)
print('Done')