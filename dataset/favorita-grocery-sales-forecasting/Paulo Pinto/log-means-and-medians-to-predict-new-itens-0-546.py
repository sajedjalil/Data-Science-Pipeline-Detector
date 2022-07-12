# -*- coding: utf-8 -*-
import pandas as pd
from datetime import timedelta
from sklearn import preprocessing

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

# Fill NaNs
train.loc[:, "unit_sales"].fillna(0, inplace=True)
train.reset_index(inplace=True) # reset index and restoring unique columns  
lastdate = train.iloc[train.shape[0]-1].date

#Load test
test = pd.read_csv("../input/test.csv", usecols=[0,2,3], dtype=dtypes)
ltest = test.shape[0]

ite = pd.read_csv('../input/items.csv')

def df_lbl_enc(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            df[c] = lbl.fit_transform(df[c])
            #print(c)
    return df

ite = df_lbl_enc(ite)
train = pd.merge(train, ite, how='left', on=['item_nbr'])
test = pd.merge(test, ite, how='left', on=['item_nbr'])

#Moving Averages
ma_is = train[["item_nbr","store_nbr","unit_sales"]].groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais226')
ma_cs = train[["class","store_nbr","unit_sales"]].groupby(['class','store_nbr'])['unit_sales'].median().to_frame('macs226')
ma_fs = train[["family","store_nbr","unit_sales"]].groupby(['family','store_nbr'])['unit_sales'].median().to_frame('mafs226')

for i in [112,56,28,14,7,3,1]:
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmpg = tmp.groupby(['item_nbr', 'store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))
    ma_is = ma_is.join(tmpg, how='left')
    tmpg = tmp.groupby(['class', 'store_nbr'])['unit_sales'].median().to_frame('macs'+str(i))
    ma_cs = ma_cs.join(tmpg, how='left')
    tmpg = tmp.groupby(['family','store_nbr'])['unit_sales'].median().to_frame('mafs'+str(i))
    ma_fs = ma_fs.join(tmpg, how='left')

del tmp,tmpg

ma_is['mais']=ma_is.median(axis=1)
ma_cs['macs']=ma_cs.min(axis=1)/2.
ma_fs['mafs']=ma_fs.min(axis=1)/4.

ma_is.reset_index(inplace=True)
ma_cs.reset_index(inplace=True)
ma_fs.reset_index(inplace=True)

test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_cs, how='left', on=['class','store_nbr'])
test = pd.merge(test, ma_fs, how='left', on=['family','store_nbr'])

test['unit_sales'] = test.mais
test.loc[test['unit_sales'].isnull(),'unit_sales'] = test.macs[test['unit_sales'].isnull()]
test.loc[test['unit_sales'].isnull(),'unit_sales'] = test.mafs[test['unit_sales'].isnull()]

test['unit_sales'] = test['unit_sales'].apply(pd.np.expm1) # restoring unit values 
test[['id','unit_sales']].to_csv('comb8.csv.gz', index=False, float_format='%.4f', compression='gzip')