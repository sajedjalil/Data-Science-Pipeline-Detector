#covid-19 global forcasting

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
dataset_y = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
y = dataset.iloc[:, 8].values
X = dataset.loc[:, dataset.columns != 'TargetValue']

#checking for null values
total = X.isnull().sum().sort_values(ascending = False)
percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

total_y = dataset_y.isnull().sum().sort_values(ascending = False)
percent_y = (dataset_y.isnull().sum()/dataset_y.isnull().count()).sort_values(ascending=False)
missing_data_y = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#dropping non-important features
X = X.drop(['Id', 'Target', 'County', 'Province_State'], axis = 1)
dataset_y = dataset_y.drop(['ForecastId', 'Target', 'County', 'Province_State'], axis = 1)

X['Date']= pd.to_datetime(X['Date']) 
dataset_y['Date']= pd.to_datetime(dataset_y['Date']) 

date = pd.to_datetime(X['Date'])
test_date = pd.to_datetime(dataset_y['Date'])

def get_month(time):
    
    return time.month

def get_day(time):
    
    return time.day
X['month'] = date.apply(lambda x: get_month(x))
X['day'] = date.apply(lambda x: get_day(x))
dataset_y['month'] = test_date.apply(lambda x: get_month(x))
dataset_y['day'] = test_date.apply(lambda x: get_day(x))

X = X.drop(['Date'], axis = 1)
dataset_y = dataset_y.drop(['Date'], axis = 1)

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_y = LabelEncoder()
X["Country_Region"] = labelencoder_X.fit_transform(X["Country_Region"])
dataset_y["Country_Region"] = labelencoder_X.fit_transform(dataset_y["Country_Region"])

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
X= sc.fit_transform(X)
dataset_y = sc.transform(dataset_y)

# Fitting XGBoost to the Training set
import xgboost as xgb
regressor = xgb.XGBRegressor(n_estimators=2000,max_depth=20)
regressor.fit(X,y)
y_pred = regressor.predict(dataset_y)
y_pred=np.around(y_pred)

pred_list = [int(x) for x in y_pred ]
output = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})

a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()

a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05'].clip(0,10000)
a['q0.5']=a['q0.5'].clip(0,10000)
a['q0.95']=a['q0.95'].clip(0,10000)
a['Id'] =a['Id']+ 1

sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.head()


sub.to_csv("submission.csv",index=False)











