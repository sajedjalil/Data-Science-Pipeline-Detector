# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import operator
import xgboost as xgb
from sklearn.model_selection import cross_val_score

seed=2017
np.random.seed(seed)

def preprocessing(x_train):
#    x_train = x_train.fillna(-1)
#    nullvar = x_train.isnull().sum()

    x_train['timestamp'] = pd.to_datetime(x_train['timestamp'])
    x_train['date'] = x_train.timestamp.dt.date
    x_train.drop(['timestamp'], axis=1, inplace=True)
    
    binary_mapping = {'yes':1,'no':0}
    x_train['culture_objects_top_25']=x_train['culture_objects_top_25'].map(binary_mapping)
    x_train['thermal_power_plant_raion']=x_train['thermal_power_plant_raion'].map(binary_mapping)
    x_train['incineration_raion']=x_train['incineration_raion'].map(binary_mapping)
    x_train['oil_chemistry_raion']=x_train['oil_chemistry_raion'].map(binary_mapping)
    x_train['radiation_raion']=x_train['radiation_raion'].map(binary_mapping)
    x_train['railroad_terminal_raion']=x_train['railroad_terminal_raion'].map(binary_mapping)
    x_train['big_market_raion']=x_train['big_market_raion'].map(binary_mapping)
    x_train['nuclear_reactor_raion']=x_train['nuclear_reactor_raion'].map(binary_mapping)
    x_train['detention_facility_raion']=x_train['detention_facility_raion'].map(binary_mapping)
    x_train['railroad_1line']=x_train['railroad_1line'].map(binary_mapping)
    
    cat_columns = x_train.select_dtypes(['object']).columns
    for col in cat_columns:
        x_train[col] = x_train[col].astype('category')
        
    x_train[cat_columns] = x_train[cat_columns].apply(lambda x:x.cat.codes)
    return x_train

def cv_out(dtrain):
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
        verbose_eval=50, show_stdv=False)
    return len(cv_output)

def XGBmodel(dtrain, num_boost_rounds):
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
    return model

def featureselection(model, x_train, x_test):
    dfscore = model.get_fscore()
    dfscore = sorted(dfscore.items(), key=operator.itemgetter(1))
    
    dfscore = pd.DataFrame(dfscore, columns=['feature','fscore'])
    dfscore['fscore'] = dfscore['fscore'].astype(int)
    
    drop = dfscore[dfscore.fscore < 10]
    
    for index,  value in drop.T.iteritems():
        x_train = x_train.drop(value.feature, axis=1)
        x_test = x_test.drop(value.feature, axis=1)
    
    return x_train, x_test
            
        
macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]

x_train = pd.read_csv('../input/train.csv')
x_test = pd.read_csv('../input/test.csv')
macro = pd.read_csv('../input/macro.csv', usecols=['timestamp'] + macro_cols)

train_corr = x_train.corr()

y_train = x_train['price_doc']
y_train_log = np.log1p(y_train)
x_train = x_train.drop('price_doc', axis=1)

x_train = x_train.merge(macro, on='timestamp')
x_test = x_test.merge(macro, on='timestamp')

x_train = preprocessing(x_train)
x_test = preprocessing(x_test)

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

#num_boost_rounds = cv_output(dtrain)
num_boost_rounds = 486 #hasil kode sebelumnya

model = XGBmodel(dtrain, num_boost_rounds)

#x_train, x_test = featureselection(model, x_train, x_test)
#
#model = XGBmodel(dtrain, num_boost_rounds)
#xb = xgb.XGBFeatureImportances()
#
y_predict =pd.DataFrame()
y_predict = model.predict(dtest)
#y_predict = np.exp(y_train_log)-1

submit=pd.DataFrame()
submit['id']=x_test.id
submit['price_doc']=y_predict
submit.to_csv('submit.csv', index=False)