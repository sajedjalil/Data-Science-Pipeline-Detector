# A Python implementation of the awesome script by Bohdan Pavlyshenko: http://tinyurl.com/jd6k2kr
# and with inspiration from Rodolfo Lomascolo's  http://tinyurl.com/z6qmxfk
#
# Author: willgraf
import numpy as np
import pandas as pd
import xgboost as xgb
import pdb

# from xgboost.sklearn import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

## --------------------------Constants-----------------------------------------
#
LAG_WEEK_VAL = 3 # set to 3 to use lagged features.
BIG_LAG = True # set to True if to use more than 1 lagged_featc

## --------------------------Functions-----------------------------------------

def get_dataframe():
    '''
    reads in a fixed column set, cleans the data, and returns a joined
    singular data set with the values to predict as null.
    '''
    print('Loading training data')
    train = pd.read_csv('../input/train.csv', 
                        usecols=['Semana','Agencia_ID','Ruta_SAK','Cliente_ID','Producto_ID','Demanda_uni_equil'])

    print('Loading test data')
    test = pd.read_csv('../input/test.csv', 
                       usecols=['Semana','Agencia_ID','Ruta_SAK','Cliente_ID','Producto_ID','id'])

    print('Merging train & test data')
    # lagged week features
    train = train.loc[train['Semana'] > LAG_WEEK_VAL,]
    train['id'] = 0
    test['Demanda_uni_equil'] = None
    train['target'] = train['Demanda_uni_equil']
    test['target'] = 0
    return pd.concat([train, test])

def create_lagged_feats(data, num_lag):
    '''
    creates a lagged demand feature based on the keys passed in
    and a number of weeks lagged. feature name: target_lNUMBER
    '''
    keys = ['Semana', 'Cliente_ID', 'Producto_ID']
    lag_feat = 'target_l' + str(num_lag)
    print('Creating lagged feature: %s' % lag_feat)
    data1 = df.loc[: , ['Cliente_ID', 'Producto_ID']]
    data1['Semana'] = df.loc[: , 'Semana'].apply(lambda x: x + 1)
    data1[lag_feat] = df.loc[: , 'target']
    data1 = pd.groupby(data1, keys).mean().reset_index()
    return pd.merge(data, data1, how='left', on=keys, left_index=False, right_index=False,
                    suffixes=('', '_lag' + str(num_lag)), copy=False)

def create_freq_feats(data, column_name):
    '''
    returns new data with a new column: column_name_freq
    which is the average weekly frequency of that value.
    '''
    freq_feat = column_name + '_freq'
    print('Creating frequency feature: %s' % freq_feat)
    freq_frame = pd.groupby(data, [column_name, 'Semana'])['target'].count().reset_index()
    freq_frame.rename(columns={'target': freq_feat}, inplace=True)
    freq_frame = pd.groupby(freq_frame, [column_name])[freq_feat].mean().reset_index()

    return pd.merge(data, freq_frame, how='left', on=[column_name], left_index=False,
                    right_index=False, suffixes=('', '_freq'), copy=False)

def build_model(df, features, model_params):
    '''
      df: dataframe to be modled
      features: column names of df that should be used in model
      params: {xgb_param_key: 'xgb_param_value'}
    '''
    mask = df['Demanda_uni_equil'].isnull()
    test = df[mask]
    train = df[~mask]
    train.loc[: , 'target'] = train.loc[: , 'target'].apply(lambda x: np.log(x + 1))

    xlf = xgb.XGBRegressor(**model_params)

    x_train, x_test, y_train, y_test = train_test_split(train[features], train['target'], test_size=0.01, random_state=1)

    xlf.fit(x_train, y_train, eval_metric='rmse', verbose=1, eval_set=[(x_test, y_test)], early_stopping_rounds=100)
    preds = xlf.predict(x_test)
    print('RMSE of log(Demanda_uni_equil[x_test]) : %s" ' % str(mean_squared_error(y_test,preds) ** 0.5))

    print('Predicting Demanda_uni_equil for Semana 10')
    data_test_10 = test.loc[test['Semana'] == 10, features]
    preds = xlf.predict(data_test_10)
    data_test_10['Demanda_uni_equil'] = np.exp(preds) - 1
    data_test_10['id'] = test.loc[test['Semana'] == 10, 'id'].astype(int).tolist()

    print('Creating lagged demand feature for Semana 11')
    data_test_lag = data_test_10[['Cliente_ID', 'Producto_ID']]
    data_test_lag['target_l1'] = data_test_10['Demanda_uni_equil']
    data_test_lag = pd.groupby(data_test_lag,['Cliente_ID','Producto_ID']).mean().reset_index()

    print('Predicting Demanda_uni_equil for Semana 11')
    data_test_11 = test.loc[test['Semana'] == 11, features.difference(['target_l1'])]
    data_test_11 = pd.merge(data_test_11, data_test_lag, how='left', on=['Cliente_ID', 'Producto_ID'],
                            left_index=False, right_index=False, sort=True, copy=False)

    data_test_11 = data_test_11.loc[: , features]#.replace(np.nan, 0, inplace=True)
    preds = xlf.predict(data_test_11)
    data_test_11['Demanda_uni_equil'] = np.exp(preds) - 1
    data_test_11['id'] = test.loc[test['Semana'] == 11, 'id'].astype(int).tolist()
    
    return pd.concat([data_test_10.loc[: , ['id', 'Demanda_uni_equil']],
                      data_test_11.loc[: , ['id', 'Demanda_uni_equil']]],
                      axis=0, copy=True)

## --------------------------Execution-----------------------------------------

## read in and lightly clean the data
df = get_dataframe()

## create lagged demand features
for i in range(1, 1 + 5):
    if not BIG_LAG and i > 1:
        break
    df = create_lagged_feats(df, num_lag=i)

df = df[df['Semana'] > 8]

## create frequnecy features
for col_name in ['Agencia_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']:
    df = create_freq_feats(df, col_name)

# df = df.replace(np.nan, 0, inplace=True)

## choose features and train model
feature_names = df.columns.difference(['id', 'target', 'Demanda_uni_equil'])
print('Data Cleaned.  Using features: %s' % feature_names)

## model parameters
xgb_params = {
    'max_depth': 10,
    'learning_rate': 0.01,
    'n_estimators': 75,
    'subsample': .85,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': True,
    'nthread': -1,
    'gamma': 0,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 0.85,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 1,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': 1,
    'missing': None,
    'seed': 1
}

## build model and write submission to file.
submission = build_model(df, feature_names, xgb_params)
submission.to_csv('../output/submission.csv', index=False)
