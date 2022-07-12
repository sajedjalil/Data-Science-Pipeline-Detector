# -*- coding: utf-8 -*-

import pandas as pd
from datetime import timedelta

dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8'}

train = pd.read_csv('../input/train.csv', usecols=[1,2,3,4], dtype=dtypes, parse_dates=['date'],
                    skiprows=range(1, 101688779) #Skip dates before 2017-01-01
                    )

train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion

# creating records for all items, in all markets on all dates
# for correct calculation of daily unit sales averages.
u_dates = train.date.unique()
u_stores = train.store_nbr.unique()
u_items = train.item_nbr.unique()
train.set_index(["date", "store_nbr", "item_nbr"], inplace=True)
train = train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=["date", "store_nbr", "item_nbr"]
    )
)

del u_dates, u_stores, u_items

train.loc[:, "unit_sales"].fillna(0, inplace=True) # fill NaNs
train.reset_index(inplace=True) # reset index and restoring unique columns  
lastdate = train.iloc[train.shape[0]-1].date

#Load test
test = pd.read_csv("../input/test.csv", usecols=[0,2,3], dtype=dtypes)
test = test.set_index(['item_nbr', 'store_nbr'])
ltest = test.shape[0]

#Moving Averages
for i in [1,3,7,14,28,56,112,224]:
    val='MA'+str(i)
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmp1 = tmp.groupby(['item_nbr', 'store_nbr'])['unit_sales'].mean().to_frame(val)
    test = test.join(tmp1, how='left')

#Median of MAs
test['unit_sales']=test.iloc[:,1:].median(axis=1)
test.loc[:, "unit_sales"].fillna(0, inplace=True)
test['unit_sales'] = test['unit_sales'].apply(pd.np.expm1) # restoring unit values 
test[['id','unit_sales']].to_csv('median_ma8.csv.gz', index=False, float_format='%.3f', compression='gzip')
