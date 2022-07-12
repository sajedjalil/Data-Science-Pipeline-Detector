# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd
import xgboost as xgb
import gc
import numpy as np
print('Loading data ...')

train = pd.read_csv('../input/train_2016_v2.csv')
prop = pd.read_csv('../input/properties_2016.csv')
sample = pd.read_csv('../input/sample_submission.csv')

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')
# drop out ouliers
df_train=df_train[ df_train.logerror > -0.38 ]
df_train=df_train[ df_train.logerror < 0.41 ]

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    if True in x_train[c]:
        x_train[c] = (x_train[c] == True)
    elif 'Y' in x_train[c]:
        x_train[c] = (x_train[c] == 'Y')
for cols in x_train.columns:
    if x_train[cols].dtype==bool:
        x_train[cols]=x_train[cols].astype(int)
    

x_train = x_train.fillna(0.0)
#x_train=np.nan_to_num(x_train)
#from sklearn.model_selection import train_test_split
#X_train,X_valid,Y_train,Y_valid=train_test_split(x_train,y_train,test_size=0.2,random_state=42)


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train,y_train)
#del X_train, Y_train; gc.collect()
df_test=sample.merge(prop,how='left',left_on='ParcelId',right_on='parcelid')
del x_train , y_train; gc.collect()
del prop; gc.collect()
x_test=df_test[train_columns]
del df_test;gc.collect()
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    if True in x_test[c]:
        x_test[c] = (x_test[c] == True)
    elif 'Y' in x_train[c]:
        x_test[c] = (x_test[c] == 'Y')
for cols in x_test.columns:
    if x_test[cols].dtype==bool:
        x_test[cols]=x_test[cols].astype(int)

#from sklearn.metrics import mean_absolute_error        
#mae=mean_absolute_error(Y_valid,clf.predict(X_valid))
#del X_valid,Y_valid;gc.collect()
#print(mae)
x_test=x_test.fillna(0.0)
y_test=clf.predict(x_test)
del x_test;gc.collect()
y_test=pd.DataFrame(y_test)
for cols in sample.columns:
    if cols != 'ParcelId':
        sample[cols]=y_test[0]
print('Writing csv...')
del y_test;gc.collect()
sample.to_csv('Starter_RandomForest.csv',index=False,float_format='%.4g')