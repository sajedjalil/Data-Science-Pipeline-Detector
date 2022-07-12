# Bimbo Python XGBoost 5 week Lag Python script adapted from R
# Rodolfo Lomascolo e-mail: r.lomascolo@rodolfolomascolo.com
# Adapted from code by Bohdan Pavlyshenko  http://tinyurl.com/jd6k2kr
# All credits for Bohdan Pavlyshenko and all bugs to me.
#
# Due to heavy usage of memory, this script will not run in kaggle.
# the script runs only with a minimum of 35GB of physical or physical+swap memory
# it can take a couple of hours to train depending on your machine, disk and memory
# I used an 8 core, 56GB of RAM Azure DS13 machine with additional 100 GB of swap

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import xgboost as xgb
from sklearn import cross_validation
import scipy.stats as stats
import gc


train = pd.read_csv('../input/train.csv') 
test = pd.read_csv('../input/test.csv')

print('Train and Test Read')

train['target'] = train['Demanda_uni_equil']
train.drop(['Demanda_uni_equil'],axis=1, inplace = True)

train['tst'] = 0
test['tst'] = 1

data = pd.concat([train,test], axis=0, copy=True)

print('Train and Test Concat')
del train
del test
gc.collect()

for i in range(1,6):
    lag = 'Lag' + str(i)
    print('Lag:',lag)
    
    data1 = data[['Semana','Cliente_ID','Producto_ID','target']]
    data1.loc[:,'Semana'] = data1['Semana'] +i
    data1 = pd.groupby(data1,['Semana','Cliente_ID','Producto_ID']).mean() 
    data1 = data1.reset_index()
    data1.rename(columns={'target': lag}, inplace=True)
    data = pd.merge(data,data1,
                    how='left',
                    left_on=['Semana','Cliente_ID','Producto_ID'], 
                    right_on=['Semana','Cliente_ID','Producto_ID'],
                    left_index=False, right_index=False, sort=True,
                    suffixes=('_x', '_y'), copy=False, )
    del data1
    gc.collect()

data['TotalLags'] = data['Lag1'] + data['Lag2']+ data['Lag3']+ data['Lag4']+ data['Lag5']
    
data=data[data['Semana']>8]  # NOW I WORK WITH WEEKS 9,10,11

nAgencia = pd.DataFrame(pd.groupby(data,['Agencia_ID','Semana'])['target'].count())
nAgencia = nAgencia.reset_index()
nAgencia.rename(columns={'target': 'nAgencia'}, inplace=True)
nAgencia = pd.DataFrame(pd.groupby(nAgencia,['Agencia_ID'])['nAgencia'].mean())
nAgencia = nAgencia.reset_index()
 

data = pd.merge(data, nAgencia, 
                            how='left',
                            left_on=['Agencia_ID'], 
                            right_on=['Agencia_ID'],
                            left_index=False, right_index=False, sort=True,
                            suffixes=('_x', '_y'), copy=False) 

del nAgencia
gc.collect()
print('merge completo nAgencia')
print(data.shape[0])

nRuta_SAK = pd.DataFrame(pd.groupby(data,['Ruta_SAK','Semana'])['target'].count())
nRuta_SAK = nRuta_SAK.reset_index()
nRuta_SAK.rename(columns={'target': 'nRuta_SAK'}, inplace=True)
nRuta_SAK = pd.DataFrame(pd.groupby(nRuta_SAK,['Ruta_SAK'])['nRuta_SAK'].mean())
nRuta_SAK = nRuta_SAK.reset_index()
 

data = pd.merge(data, nRuta_SAK, 
                            how='left',
                            left_on=['Ruta_SAK'], 
                            right_on=['Ruta_SAK'],
                            left_index=False, right_index=False, sort=True,
                            suffixes=('_x', '_y'), copy=False) 

del nRuta_SAK
gc.collect()
print('merge completo nRuta_SAK')
print(data.shape[0])

nCliente_ID = pd.DataFrame(pd.groupby(data,['Cliente_ID','Semana'])['target'].count())
nCliente_ID = nCliente_ID.reset_index()
nCliente_ID.rename(columns={'target': 'nCliente_ID'}, inplace=True)
nCliente_ID = pd.DataFrame(pd.groupby(nCliente_ID,['Cliente_ID'])['nCliente_ID'].mean())
nCliente_ID = nCliente_ID.reset_index()
 

data = pd.merge(data, nCliente_ID, 
                            how='left',
                            left_on=['Cliente_ID'], 
                            right_on=['Cliente_ID'],
                            left_index=False, right_index=False, sort=True,
                            suffixes=('_x', '_y'), copy=False) 

del nCliente_ID
gc.collect()
print('merge completo nCliente_ID')
print(data.shape[0])

nProducto_ID = pd.DataFrame(pd.groupby(data,['Producto_ID','Semana'])['target'].count())
nProducto_ID = nProducto_ID.reset_index()
nProducto_ID.rename(columns={'target': 'nProducto_ID'}, inplace=True)
nProducto_ID = pd.DataFrame(pd.groupby(nProducto_ID,['Producto_ID'])['nProducto_ID'].mean())
nProducto_ID = nProducto_ID.reset_index()
 

data = pd.merge(data, nProducto_ID, 
                            how='left',
                            left_on=['Producto_ID'], 
                            right_on=['Producto_ID'],
                            left_index=False, right_index=False, sort=True,
                            suffixes=('_x', '_y'), copy=False) 

del nProducto_ID
gc.collect()
print('merge completo nProducto_ID')
print(data.shape[0])

data.replace(np.nan,0, inplace=True)

train = data[data['tst']==0]
predict = data[data['tst']==1]

train['target'] = np.log(train['target'] + 1)
#train2 = train.sample(n=1000000)   <-- another possible reduction of data for fast testing
train2 = train
y = train['target']
X = train[[  'Agencia_ID','Canal_ID','Cliente_ID','Producto_ID','Ruta_SAK',
             'Lag1','Lag2','Lag3','Lag4','Lag5','TotalLags',
             'nAgencia','nRuta_SAK','nCliente_ID','nProducto_ID']]

print(X.shape, y.shape)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1729)
print(X_train.shape, X_test.shape)


xlf = xgb.XGBRegressor(max_depth=10, 
                        learning_rate=0.1, 
                        n_estimators=10, 
                        silent=True, 
                        objective='reg:linear', 
                        nthread=-1, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None)

xlf.fit(X_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, y_test)],early_stopping_rounds=100)

# calculate the auc score
preds = xlf.predict(X_test)

print('\nRoot Mean Square error of Log(Demanda) X_test" ', mean_squared_error(y_test,preds)**0.5)


# GENERATE PREDICTION FOR 10-S11 FOR KAGGLE SUBMISSION

print('Make prediction for the 10th week')
data_test1=predict[predict['Semana']==10]
ids_10 =data_test1['id'] 
data_test1 = data_test1[['Agencia_ID','Canal_ID','Cliente_ID','Producto_ID','Ruta_SAK',
                         'Lag1','Lag2','Lag3','Lag4','Lag5','TotalLags',
                         'nAgencia','nRuta_SAK','nCliente_ID','nProducto_ID']] 

pred = xlf.predict(data_test1)
res=np.exp(pred)-1

# Create lagged values of target variable which will be used as a feature for the 11th week prediction 
print('Create lagged values of target variable which will be used as a feature for the 11th week prediction')
data_test_lag1=data_test1[['Cliente_ID','Producto_ID']]
data_test_lag1['targetl1']=res
data_test_lag1 = pd.groupby(data_test_lag1,['Cliente_ID','Producto_ID']).mean() 
data_test_lag1 = data_test_lag1.reset_index()
data_test_lag1.rename(columns={'targetl1': 'Lag1'}, inplace=True)

# Save the predictions for first week
data_test1['Demanda_uni_equil'] = res
data_test1['id'] = ids_10.astype(int).tolist()

# Make prediction for the 11th week
print('Make prediction for the 11th week')

data_test2=predict[predict['Semana']==11]
ids_11 =data_test2['id']
data_test2 = data_test2[[  'Agencia_ID','Canal_ID','Cliente_ID','Producto_ID','Ruta_SAK',
             'Lag2','Lag3','Lag4','Lag5','TotalLags',
             'nAgencia','nRuta_SAK','nCliente_ID','nProducto_ID']] 

data_test2 = pd.merge(data_test2, data_test_lag1, 
                            how='left',
                            left_on=['Cliente_ID','Producto_ID'], 
                            right_on=['Cliente_ID','Producto_ID'],
                            left_index=False, right_index=False, sort=True,
                            suffixes=('_x', '_y'), copy=False)

data_test2['TotalLags'] = data_test2['Lag1'] + data_test2['Lag2']+ data_test2['Lag3']+ data_test2['Lag4']+ data_test2['Lag5']

data_test2 = data_test2[['Agencia_ID','Canal_ID','Cliente_ID','Producto_ID','Ruta_SAK',
                         'Lag1','Lag2','Lag3','Lag4','Lag5','TotalLags',
                         'nAgencia','nRuta_SAK','nCliente_ID','nProducto_ID']] 

data_test2.replace(np.nan,0, inplace=True)
pred = xlf.predict(data_test2)
res=np.exp(pred)-1

# Save the predictions for Second week
data_test2['Demanda_uni_equil'] = res
data_test2['id'] = ids_11.astype(int).tolist()


submit = pd.concat([data_test1[['id','Demanda_uni_equil']],data_test2[['id','Demanda_uni_equil']]], axis=0, copy=True)

print('Saving Predictions file')
submit.to_csv('Predictions/S10-S11-Prediction.csv', index=False)