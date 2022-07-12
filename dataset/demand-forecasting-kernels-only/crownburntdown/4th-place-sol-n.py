 #prompt:
##You are given 5 years of store-item sales data, 
#and asked to predict 3 months of sales for 50 different items 
#at 10 different stores.
##
##What's the best way to deal with seasonality? Should stores 
#be modeled separately, or can you pool them together? Does deep 
#learning work better than ARIMA? Can either beat xgboost?
##
##This is a great competition to explore different models and
# improve your skills in forecasting.


import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
from sklearn.model_selection import KFold
#from scipy.stats import skew,kurtosis,gmean,hmean
import matplotlib.pyplot as plt
import os

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')
#date - Date of the sale data. There are no holiday effects or store closures.
#store - Store ID
#item - Item ID
#sales - Number of items sold at a particular store on a particular date

#define smape
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

#concatenate train and test
cols=list(train.columns)
x = pd.concat([train,test],axis=0).reset_index(drop=True)
x = x.loc[:,cols]

#index to timestamp, add date feats
x.index=pd.to_datetime(x.date)
x.drop('date',axis=1,inplace=True)

x['year'] =  x.index.year - min(x.index.year) + 1
x['month'] = x.index.month
x['weekday'] = x.index.weekday

x = x[x.year > 1]

month_smry= (
        ( x.groupby(['month']).agg([np.nanmean]).sales - np.nanmean(x.sales) ) / 
        np.nanmean(x.sales)
).rename(columns={'nanmean':'month_mod'})
x=x.join(month_smry,how='left',on='month')

year_smry= (
        ( x.groupby(['year']).agg([np.nanmean]).sales - np.nanmean(x.sales) ) / 
        np.nanmean(x.sales)
).rename(columns={'nanmean':'year_mod'})
#calculate CAGR
#x.groupby(['year']).agg([np.nanmean,np.nanstd]).sales.plot()
CAGR = (x[x.year==5].groupby(['store','item']).agg(np.nanmean).sales /
x[x.year==2].groupby(['store','item']).agg(np.nanmean).sales )**(1/4)-1
print((np.mean(CAGR),np.std(CAGR)))
#(0.062168562201314385, 0.0023445325283927287)
year_smry.loc[6,:] =  np.mean(CAGR)*3
x=x.join(year_smry,how='left',on='year')

weekday_smry= (
        ( x.groupby(['weekday']).agg([np.nanmean]).sales - np.nanmean(x.sales) ) / 
        np.nanmean(x.sales)
).rename(columns={'nanmean':'weekday_mod'})
x=x.join(weekday_smry,how='left',on='weekday')

store_item_smry= (
        ( x.groupby(['store','item']).agg([np.nanmean]).sales - np.nanmean(x.sales) ) / 
        np.nanmean(x.sales)
).rename(columns={'nanmean':'store_item_mod'})
x=x.join(store_item_smry,how='left',on=['store','item'])

x['smry_product']=np.product(x.loc[:,['month_mod','year_mod','weekday_mod','store_item_mod',]]+1,axis=1)


x['sales_mod_pred']=np.round(x.smry_product*np.round(np.nanmean(x.sales),1))

print(smape(x.sales,x.sales_mod_pred))
#12.202514491900835

print(smape(x.sales[x.month < 4],x.sales_mod_pred[x.month < 4]))
#13.6915833228265

plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(x.sales,x.sales_mod_pred,s=2)
plt.axis('equal')
plt.show()
plt.savefig('graph.png')

plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(x.sales[x.month < 4],x.sales_mod_pred[x.month < 4],s=2)
plt.axis('equal')
plt.show()
plt.savefig('graph.png')



# submission
sample['sales'] = x[x.year==6].sales_mod_pred.reset_index(drop=True)
sample.to_csv('submittal.csv', index= False)