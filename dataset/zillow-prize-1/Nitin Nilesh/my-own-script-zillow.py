#All import statements
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import gc

print("All packages Imported")

#Load All the Datasets
print("Loading Datasets")
train = pd.read_csv('../input/train_2016_v2.csv')
properties = pd.read_csv('../input/properties_2016.csv',low_memory=False)
sample = pd.read_csv('../input/sample_submission.csv')
print("Datasets Loaded")

print('Making own feature set')
for c, dtype in zip(properties.columns, properties.dtypes):
    if dtype == np.float64:
        properties[c] = properties[c].astype(np.float32)

properties['N_LivingAreaError'] = properties['calculatedfinishedsquarefeet'] / properties[
        'finishedsquarefeet12']
        
ulimit = np.percentile(train.logerror.values, 99)
llimit = np.percentile(train.logerror.values, 1)
train[ train.logerror < llimit ] = llimit
train[ train.logerror > ulimit ] = ulimit

col = "taxamount"
ulimit = np.percentile(properties[col].values, 99.5)
llimit = np.percentile(properties[col].values, 0.5)
properties[col].loc[properties[col]>ulimit] = ulimit
properties[col].loc[properties[col]<llimit] = llimit

properties['bedroomcnt'].loc[properties['bedroomcnt']>7] = 7

col = "calculatedfinishedsquarefeet"
ulimit = np.percentile(properties[col].values, 99.5)
llimit = np.percentile(properties[col].values, 0.5)
properties[col].loc[properties[col]>ulimit] = ulimit
properties[col].loc[properties[col]<llimit] = llimit

col = "finishedsquarefeet12"
ulimit = np.percentile(properties[col].values, 99.5)
llimit = np.percentile(properties[col].values, 0.5)
properties[col].loc[properties[col]>ulimit] = ulimit
properties[col].loc[properties[col]<llimit] = llimit

df_train = train.merge(properties, how='left', on='parcelid')
for c in properties.dtypes[properties.dtypes == object].index.values:
    properties[c] = (properties[c] == True)
    
Y_train = np.array(train["logerror"])
y_mean = np.mean(Y_train)
keep_cols = ['structuretaxvaluedollarcnt','calculatedfinishedsquarefeet','lotsizesquarefeet'
            ,'taxvaluedollarcnt','taxamount','bathroomcnt','landtaxvaluedollarcnt',
            'bedroomcnt','finishedsquarefeet12','N_LivingAreaError','heatingorsystemtypeid',
            'lotsizesquarefeet','longitude','latitude','yearbuilt']
X_train = df_train[keep_cols]

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(properties, on='parcelid', how='left')
X_test = df_test[X_train.columns]
print(X_train.shape, Y_train.shape, X_test.shape)
print("Finished Making features")

#Clean Data
print('cleaning Data')
imputer= Imputer(strategy='most_frequent')
imputer.fit(X_train.iloc[:, :])
X_train = imputer.transform(X_train.iloc[:,:])
print("X_train done")
for c in X_test.columns:
        X_test[c]=X_test[c].fillna(-1)
print('filled missing values')
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Classifying Data
print("Training Data")
'''len_x=int(X_train.shape[1])
print("len_x is:",len_x)
classifier = Sequential()
classifier.add(Dense(units = 14 , kernel_initializer = 'normal', activation = 'relu', input_dim = len_x))
classifier.add(Dense(units = 7, kernel_initializer = 'normal', activation = 'relu'))
classifier.add(Dense(units = 3, kernel_initializer = 'normal', activation = 'relu'))
classifier.add(Dense(1, kernel_initializer='normal'))
classifier.compile(loss='mae', optimizer='rmsprop', metrics=['mae', 'accuracy'])

classifier.fit(np.array(X_train), np.array(Y_train), batch_size = 10, epochs = 10)

print("X_test.shape:",X_test.shape)
y_pred_ann = classifier.predict(X_test)

y_pred = y_pred_ann.flatten()'''
xgb_params = {
    'eta': 0.007,
    'max_depth': 7,
    'subsample': 0.60,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 5.0,
    'alpha': 0.65,
    'colsample_bytree': 0.5,
    'base_score': y_mean,
    'silent': 1
}
dtrain = xgb.DMatrix(X_train, Y_train)
dtest = xgb.DMatrix(X_test)

early_stopping_rounds = round( 1 / xgb_params['eta'] )
num_boost_rounds = round( 20 / xgb_params['eta'] )
# cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   nfold=5,
                   num_boost_round=num_boost_rounds,
                   early_stopping_rounds=early_stopping_rounds,
                   verbose_eval=10, 
                   show_stdv=False
                  )
num_boost_rounds = len(cv_result)
print(num_boost_rounds)
# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
pred = model.predict(dtest)
y_pred=[]

for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

#Writing data to csv
print("Writing data to csv file")
output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output.to_csv('Using_Keras_Regressor{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
print( "\nFinished!" )

