# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

 
df1=pd.read_csv("../input/historical_transactions.csv", sep=',', lineterminator='\n')#,nrows=10000000)
#df2=pd.read_csv("../input/merchants.csv", sep=',', lineterminator='\n')

print(df1[['card_id','month_lag','purchase_amount']])
df1_1 =df1#.query('card_id=="C_ID_5037ff576e"')
df1_1.purchase_amount=df1_1.purchase_amount.abs()
df1_1.month_lag=df1_1.month_lag.abs()

mtp=df1_1.groupby(['card_id','month_lag']) [['purchase_amount']].mean()
mtp['lag_0'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=0)
mtp['lag_1'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=1)
mtp['lag_2'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=2)
mtp['lag_3'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=3)
mtp['lag_4'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=4)
mtp['lag_5'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=5)
mtp['lag_6'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=6)
mtp['lag_7'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=7)
mtp['lag_8'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=8)
mtp['lag_9'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=9)
mtp['lag_10'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=10)
mtp['lag_11'] = mtp.groupby(['card_id'])['purchase_amount'].shift(periods=11)
mtp['lag_12'] =mtp.groupby(['card_id'])['purchase_amount'].shift(periods=12)
#mtp['mavg'] =mtp.purchase_amount.rolling(window=4).mean()
#mtp=mtp.query('month_lag==0')

def  fillNafunc(df):
    for x in range(13):
        col_name="lag_"+str(x)
        df[col_name].fillna(value=0, inplace=True)
fillNafunc(mtp)



"""
mtp.lag_1.fillna(value=1, inplace=True)
mtp.lag_4.fillna(value=1, inplace=True)
mtp.lag_8.fillna(value=1, inplace=True)
mtp.lag_8.fillna(value=1, inplace=True)
mtp.lag_12.fillna(value=1, inplace=True)
"""
#GETTING LAST RECORD FOR EACH CARD_ID
mtp=mtp.sort_values('month_lag').groupby('card_id').tail(1)
print(mtp[['lag_0','lag_1','lag_4','lag_8','lag_12']])


df3=pd.read_csv("../input/train.csv", sep=',', lineterminator='\n')
#print(df3.shape)
result = df3.merge(mtp, on=['card_id'],suffixes=("_df", "_res"))
#print(result.shape)
print("result card_id",result.card_id.is_unique)
#print(result.columns.values)
"""
['first_active_month' 'card_id' 'feature_1' 'feature_2' 'feature_3'
 'target' 'purchase_amount' 'lag_0' 'lag_1' 'lag_4' 'lag_8' 'lag_12']
"""


df1_2=df1_1[["card_id","merchant_id"]]
df1_2=df1_2.drop_duplicates()






#merge with merchant dataset

df_m=pd.read_csv("../input/merchants.csv", sep=',', lineterminator='\n')
df_m2=df_m[["merchant_id","avg_sales_lag3","avg_purchases_lag3","active_months_lag3","avg_sales_lag6","avg_purchases_lag6","active_months_lag6",
"avg_sales_lag12","avg_purchases_lag12","active_months_lag12"]]
df_m2 = df_m2.sort_values('merchant_id', ascending=False)
df_m2=df_m2.drop_duplicates(subset ="merchant_id",keep='first')
#print("df_m merchant_id",df_m2.merchant_id.is_unique)
#print(df_m2.head())


print("df1_2 shape",df1_2.shape)
result2=df1_2.merge( df_m2, on=['merchant_id'],suffixes=("_xx", "_yy") )
result2 = result2.sort_values('card_id', ascending=False)
result2=result2.drop_duplicates(subset ="card_id",keep='first')
print("check card_id uniquer",result2.card_id.is_unique)
print(result2.columns.values)
""
['card_id' 'merchant_id' 'avg_sales_lag3' 'avg_purchases_lag3'
 'active_months_lag3' 'avg_sales_lag6' 'avg_purchases_lag6'
 'active_months_lag6' 'avg_sales_lag12' 'avg_purchases_lag12'
 'active_months_lag12']
""

result3 = result.merge(result2, on=['card_id'],suffixes=("_r1", "_r2"))
print(result3.columns.values)
print("result3 check card_id uniquer",result3.card_id.is_unique)
"""
['first_active_month' 'card_id' 'feature_1' 'feature_2' 'feature_3'
 'target' 'purchase_amount' 'lag_0' 'lag_1' 'lag_2' 'lag_3' 'lag_4'
 'lag_5' 'lag_6' 'lag_7' 'lag_8' 'lag_9' 'lag_10' 'lag_11' 'lag_12'
 'merchant_id' 'avg_sales_lag3' 'avg_purchases_lag3' 'active_months_lag3'
 'avg_sales_lag6' 'avg_purchases_lag6' 'active_months_lag6'
 'avg_sales_lag12' 'avg_purchases_lag12' 'active_months_lag12']
"""


def  fillNaCol(df):
    for y in  df.columns:
        df[y].fillna(value=1, inplace=True)

#dealing  with null and inf values
fillNaCol(result3)
result3 = result3.replace([np.inf, -np.inf], 1)


#lets make guess over result dataframe
lister_4=['feature_1', 'feature_2', 'feature_3',
'lag_1','lag_2','lag_3' ,'lag_4' ,'lag_5',
'lag_6','lag_7' ,'lag_8' ,'lag_9','lag_10','lag_11','lag_12',
'avg_sales_lag3','avg_purchases_lag3',
'avg_sales_lag6', 'avg_purchases_lag6', 
'avg_sales_lag12' ,'avg_purchases_lag12']

lister_5=['target']

print("problem column")
print(result3.avg_purchases_lag3)


#at first split into  train and test dataset
X_train, X_test, y_train, y_test= train_test_split(result3[lister_4],result3[lister_5], test_size=0.3,
random_state=42)
#result3 = result3.drop("target", axis=1)

#col_mask=X_test.isnull().any(axis=0)
#print(col_mask)
print(pd.isnull(X_test).sum() > 0)
print(pd.isnull(X_train).sum() > 0)
#print(X_test.info())


#scale numeric values
scaler=StandardScaler()
X_train_scale=scaler.fit_transform(X_train)
X_test_scale=scaler.transform(X_test)



param_est=[40]
param_eta=[0.1]
param_dep=[6]
for k in param_est:
    
    for  z in param_eta:
        
        for  d in param_dep:
            #checking  RMSE
            xgb=XGBRegressor(n_estimators =k,max_depth = d,eta=z,random_state=43 )
            xgb.fit(X_train_scale,y_train)
            rmse = sqrt(mean_squared_error(y_test, xgb.predict(X_test_scale)))
            print(k," ",z," ",d)
            print("RMSE RESULT ",rmse)



df3_test=pd.read_csv("../input/test.csv", sep=',', lineterminator='\n')
result_test = df3_test.merge(mtp, on=['card_id'],suffixes=("_df", "_res"))
result3_test = result_test.merge(result2, on=['card_id'],suffixes=("_r1", "_r2"))

fillNaCol(result3_test)
result3_test = result3_test.replace([np.inf, -np.inf], 1)

X_test_scale=scaler.transform(result3_test[lister_4])


predictions = xgb.predict(X_test_scale)
my_submission = pd.DataFrame( { 'card_id': result3_test.card_id,'target': predictions } )
# you could use any filename. We choose submission here
my_submission.to_csv('sample_submission.csv', index=False)
print("Writing complete")
##


""" 
xgb.fit(predictor_var,outcome_var)
predictions = model.predict(predictor_var_test)

my_submission = pd.DataFrame( { 'Id': dft2.Id,'SalePrice': predictions } )
# you could use any filename. We choose submission here
my_submission.to_csv('sample_submission.csv', index=False)
print("Writing complete")


 

def convertbin(dfx):
    dfx['authorized_flag']=dfx['authorized_flag'].map(lambda authorized_flag: 1 if authorized_flag=='Y' else 0)
    return dfx
    
df1_1=convertbin(df1_1)

print(df1_1.columns.values)
"""
['authorized_flag' 'card_id' 'city_id' 'category_1' 'installments'
 'category_3' 'merchant_category_id' 'merchant_id' 'month_lag'
 'purchase_amount' 'purchase_date' 'category_2' 'state_id' 'subsector_id']
"""

"""





