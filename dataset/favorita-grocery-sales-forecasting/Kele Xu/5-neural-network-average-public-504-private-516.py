# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import os
import gc
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def create_transactions_2017():
    transactions = pd.read_csv("../input/transactions.csv",
                              parse_dates=["date"])
    transactions_2017 =  transactions[transactions.date.isin(
        pd.date_range("2017-04-01", periods=7 * 11 + 63))].copy()

    transactions_2017 = transactions_2017.set_index(['store_nbr','date']).unstack()['transactions']
    return transactions_2017

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)
    
df_test = pd.read_csv(
    "../input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]

del df_train

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)

promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)

df_2017.columns = df_2017.columns.get_level_values(1)
items = pd.read_csv(
    "../input/items.csv",
).set_index("item_nbr")

items = items.reindex(df_2017.index.get_level_values(1))

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def create_feat_count(df,feat):
    store = pd.read_csv("../input/stores.csv")
    item = pd.read_csv("../input/items.csv")
    data_or = pd.DataFrame(df.stack(),columns=['unit_sales']).reset_index()

    data_or = pd.merge(data_or,item,how='left',on=['item_nbr'])
    data_or = pd.merge(data_or,store,how='left',on=['store_nbr'])
    data_or = data_or.groupby([feat,'date']).agg({'unit_sales':'sum'})['unit_sales'].unstack()
    X_feat = pd.DataFrame({
            "mean_feat":data_or.mean(axis=1).values,
            "max_feat":data_or.max(axis=1).values,
            "std_feat":data_or.mean(axis=1).values,
            "min_feat":data_or.min(axis=1).values,
            "skew_feat":data_or.skew(axis=1).values,
            "kurt_feat":data_or.kurt(axis=1).values,
        })
#     data_or.columns = range(data_or.shape[1]) 
    X_feat = X_feat.add_suffix("_%s"%(feat))   
    X_feat[feat] = data_or.index.tolist()
    return X_feat

def create_feat_data(df_2017,feat):
    feat_data = df_2017[[feat,'date','unit_sales']].groupby([feat,'date']).agg({'unit_sales':'mean'})['unit_sales'].unstack()
    return feat_data

def generate_feat(df,t2017,feat):
    data_or = pd.DataFrame(df.stack(),columns=['unit_sales']).reset_index()
    feat_data = create_feat_data(data_or,feat)
    X = pd.DataFrame({
        "mean_3_2017": get_timespan(feat_data, t2017, 3, 3).mean(axis=1).values,
        "std_3_2017": get_timespan(feat_data, t2017, 3, 3).std(axis=1).values, 
        "max_3_2017": get_timespan(feat_data, t2017, 3, 3).max(axis=1).values,   
        "min_3_2017": get_timespan(feat_data, t2017, 3, 3).min(axis=1).values, 
          
        "mean_7_2017": get_timespan(feat_data, t2017, 7, 7).mean(axis=1).values,
        "std_7_2017": get_timespan(feat_data, t2017, 7, 7).std(axis=1).values,
        "max_7_2017": get_timespan(feat_data, t2017, 7, 7).max(axis=1).values,
        "min_7_2017": get_timespan(feat_data, t2017, 7, 7).min(axis=1).values,
        "null_7_2017":np.sum(get_timespan(feat_data, t2017, 7, 7).isnull(),axis=1).values, 
        
        "mean_14_2017": get_timespan(feat_data, t2017, 14, 14).mean(axis=1).values,
        "std_14_2017": get_timespan(feat_data, t2017, 14, 14).std(axis=1).values,
        "max_14_2017": get_timespan(feat_data, t2017, 14, 14).max(axis=1).values,
        "min_14_2017": get_timespan(feat_data, t2017, 14, 14).min(axis=1).values,
                   
    })
    X = X.add_suffix("_%s"%(feat))
    X[feat] = feat_data.index.values
    return X

def create_feats_combine(df,feats, minus, periods):
    feats.append("date")
    store = pd.read_csv("../input/stores.csv")
    item = pd.read_csv("../input/items.csv")
    data_or = pd.DataFrame(df.stack(),columns=['unit_sales']).reset_index()
    data_or = pd.merge(data_or,item,how='left',on=['item_nbr'])
    data_or = pd.merge(data_or,store,how='left',on=['store_nbr'])
    data_or = data_or.groupby(feats).agg({'unit_sales':'sum'})['unit_sales'].unstack()
    X_feat = pd.DataFrame({
            "mean_feat":data_or.mean(axis=1).values,
            "max_feat":data_or.max(axis=1).values,
            "std_feat":data_or.mean(axis=1).values,
            "min_feat":data_or.min(axis=1).values,
            "skew_feat":data_or.skew(axis=1).values,
            "kurt_feat":data_or.kurt(axis=1).values,
            "median_feat":data_or.median(axis=1).values,
        })
    feats.remove('date')
    X_feat = X_feat.add_suffix("_%s_%s_%d_%d"%(feats[0],feats[1],minus,periods))   
    X_feat.index = data_or.index
    return X_feat

def get_timespan_featscombine(df, dt, minus, periods,feats):
    df_ts = get_timespan(df, dt, minus, periods)
    analy_feat = create_feats_combine(df_ts,feats, minus, periods)
    return analy_feat

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "std_3_2017": get_timespan(df_2017, t2017, 3, 3).std(axis=1).values, 
        "max_3_2017": get_timespan(df_2017, t2017, 3, 3).max(axis=1).values,   
        "min_3_2017": get_timespan(df_2017, t2017, 3, 3).min(axis=1).values, 
        
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "std_7_2017": get_timespan(df_2017, t2017, 7, 7).std(axis=1).values,
        "max_7_2017": get_timespan(df_2017, t2017, 7, 7).max(axis=1).values,
        "min_7_2017": get_timespan(df_2017, t2017, 7, 7).min(axis=1).values,
        "skew_7_2017": get_timespan(df_2017, t2017, 7, 7).skew(axis=1).values,
        "kurt_7_2017": get_timespan(df_2017, t2017, 7, 7).kurt(axis=1).values,
        
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "std_14_2017": get_timespan(df_2017, t2017, 14, 14).std(axis=1).values,
        "max_14_2017": get_timespan(df_2017, t2017, 14, 14).max(axis=1).values,
        "skew_14_2017": get_timespan(df_2017, t2017, 14, 14).skew(axis=1).values,
        "kurt_14_2017": get_timespan(df_2017, t2017, 14, 14).kurt(axis=1).values,
        
        "mean_21_2017": get_timespan(df_2017, t2017, 21, 21).mean(axis=1).values,
        "std_21_2017": get_timespan(df_2017, t2017, 21, 21).std(axis=1).values,
        "max_21_2017": get_timespan(df_2017, t2017, 21, 21).max(axis=1).values,
        "skew_21_2017": get_timespan(df_2017, t2017, 21, 21).skew(axis=1).values,
        "kurt_21_2017": get_timespan(df_2017, t2017, 21, 21).kurt(axis=1).values,
                
        "mean_28_2017": get_timespan(df_2017, t2017, 28, 28).mean(axis=1).values,
        "std_28_2017": get_timespan(df_2017, t2017, 28, 28).std(axis=1).values,
        "max_28_2017": get_timespan(df_2017, t2017, 28, 28).max(axis=1).values,
        "skew_28_2017": get_timespan(df_2017, t2017, 28, 28).skew(axis=1).values,
        "kurt_28_2017": get_timespan(df_2017, t2017, 28, 28).kurt(axis=1).values,
            
        "mean_35_2017": get_timespan(df_2017, t2017, 35, 35).mean(axis=1).values,
        "std_35_2017": get_timespan(df_2017, t2017, 35, 35).std(axis=1).values,
        "max_35_2017": get_timespan(df_2017, t2017, 35, 35).max(axis=1).values,
        "skew_35_2017": get_timespan(df_2017, t2017, 35, 35).skew(axis=1).values,
        "kurt_35_2017": get_timespan(df_2017, t2017, 35, 35).kurt(axis=1).values,
            
        "mean_42_2017": get_timespan(df_2017, t2017, 42, 42).mean(axis=1).values,
        "std_42_2017": get_timespan(df_2017, t2017, 42, 42).std(axis=1).values,
        "max_42_2017": get_timespan(df_2017, t2017, 42, 42).max(axis=1).values,
        "skew_42_2017": get_timespan(df_2017, t2017, 42, 42).skew(axis=1).values,
        "kurt_42_2017": get_timespan(df_2017, t2017, 42, 42).kurt(axis=1).values,            
            
        "mean_49_2017": get_timespan(df_2017, t2017, 49, 49).mean(axis=1).values,
        "std_49_2017": get_timespan(df_2017, t2017, 49, 49).std(axis=1).values,
        "max_49_2017": get_timespan(df_2017, t2017, 49, 49).max(axis=1).values,
        "skew_49_2017": get_timespan(df_2017, t2017, 49, 49).skew(axis=1).values,
        "kurt_49_2017": get_timespan(df_2017, t2017, 49, 49).kurt(axis=1).values,
        
        "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
        "std_60_2017": get_timespan(df_2017, t2017, 60, 60).std(axis=1).values,
        "max_60_2017": get_timespan(df_2017, t2017, 60, 60).max(axis=1).values,
        "skew_60_2017": get_timespan(df_2017, t2017, 60, 60).skew(axis=1).values,
        "kurt_60_2017": get_timespan(df_2017, t2017, 60, 60).kurt(axis=1).values,    
        
        "null_14_2017":np.sum(get_timespan(df_2017, t2017, 14, 14).isnull(),axis=1).values, 
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "store_nbr":df_2017.reset_index("store_nbr")['store_nbr'].values,
        "item_nbr":df_2017.reset_index("item_nbr")['item_nbr'].values
   
    })
    X['mean_3_7_diff'] = X['mean_3_2017'] - X['mean_7_2017']
    X['mean_7_14_diff'] = X['mean_7_2017'] - X['mean_14_2017']
    X['mean_14_28_diff'] = X['mean_14_2017'] - X['mean_28_2017']
    X['mean_28_35_diff'] = X['mean_28_2017'] - X['mean_35_2017']
    X['mean_35_42_diff'] = X['mean_35_2017'] - X['mean_42_2017']
    X['mean_42_49_diff'] = X['mean_42_2017'] - X['mean_49_2017']
    
    item = pd.read_csv("../input/items.csv")
    store = pd.read_csv("../input/stores.csv")
    
    combin_feat1 = ['store_nbr','family']
    combin_feat2 = ['item_nbr','city']
    sf_feat_140 = get_timespan_featscombine(df_2017,t2017,140,140,combin_feat1).reset_index()
    sf_feat_60 = get_timespan_featscombine(df_2017,t2017,60,60,combin_feat1).reset_index()
    sf_feat_30 = get_timespan_featscombine(df_2017,t2017,30,30,combin_feat1).reset_index()
    sf_feat_21 = get_timespan_featscombine(df_2017,t2017,21,21,combin_feat1).reset_index()
    sf_feat_14 = get_timespan_featscombine(df_2017,t2017,14,14,combin_feat1).reset_index()
    sf_feat_7 = get_timespan_featscombine(df_2017,t2017,7,7,combin_feat1).reset_index()
    sf_feat_3 = get_timespan_featscombine(df_2017,t2017,3,3,combin_feat1).reset_index()
    
    ci_feat_140 = get_timespan_featscombine(df_2017,t2017,140,140,combin_feat2).reset_index()
    ci_feat_60 = get_timespan_featscombine(df_2017,t2017,60,60,combin_feat2).reset_index()
    ci_feat_30 = get_timespan_featscombine(df_2017,t2017,30,30,combin_feat2).reset_index()
    ci_feat_21 = get_timespan_featscombine(df_2017,t2017,21,21,combin_feat2).reset_index()
    ci_feat_14 = get_timespan_featscombine(df_2017,t2017,14,14,combin_feat2).reset_index()
    ci_feat_7 = get_timespan_featscombine(df_2017,t2017,7,7,combin_feat2).reset_index()
    ci_feat_3 = get_timespan_featscombine(df_2017,t2017,3,3,combin_feat2).reset_index()
    
    store_feat = generate_feat(df_2017,t2017,'store_nbr')
    item_feat = generate_feat(df_2017,t2017,'item_nbr')
    X = pd.merge(X,store_feat,how='left',on=['store_nbr'])
    X = pd.merge(X,item_feat,how='left',on=['item_nbr'])
    X = pd.merge(X,item,how='left',on=['item_nbr'])
    X = pd.merge(X,store,how='left',on=['store_nbr'])
    
    
    X = pd.merge(X,sf_feat_140,how='left',on=combin_feat1)
    X = pd.merge(X,sf_feat_60,how='left',on=combin_feat1)
    X = pd.merge(X,sf_feat_30,how='left',on=combin_feat1)
    X = pd.merge(X,sf_feat_21,how='left',on=combin_feat1)
    X = pd.merge(X,sf_feat_14,how='left',on=combin_feat1)
    X = pd.merge(X,sf_feat_7,how='left',on=combin_feat1)
    X = pd.merge(X,sf_feat_3,how='left',on=combin_feat1)
    
    X = pd.merge(X,ci_feat_140,how='left',on=combin_feat2)
    X = pd.merge(X,ci_feat_60,how='left',on=combin_feat2)
    X = pd.merge(X,ci_feat_30,how='left',on=combin_feat2)
    X = pd.merge(X,ci_feat_21,how='left',on=combin_feat2)
    X = pd.merge(X,ci_feat_14,how='left',on=combin_feat2)
    X = pd.merge(X,ci_feat_7,how='left',on=combin_feat2)
    X = pd.merge(X,ci_feat_3,how='left',on=combin_feat2)
    X['fc'] = X['store_nbr'].astype(str) + "_" + X['item_nbr'].astype(str)
 
    unit_day = get_timespan(df_2017, t2017, 49, 49)
    unit_day.columns.name = None
    unit_day.columns = range(unit_day.shape[1])
    unit_day = unit_day.add_suffix("_day_unit")
    unit_day = unit_day.reset_index()

    
    X = pd.merge(X,unit_day,how='left',on=['store_nbr','item_nbr'])
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
    
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)

    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values

        return X, y
    return X

t2017 = date(2017, 6, 7)

X_l, y_l = [], []
for i in range(7):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
    print(("create feature %s"%(str(i))))
    
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)

del X_l, y_l


X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

item_nbr_list = X_train['item_nbr'].unique().tolist()
def item_nbr_transform(item_nbr,item_nbr_list):
    if item_nbr in item_nbr_list:
        return item_nbr
    else:
        return 0

X_test['item_nbr'] = X_test['item_nbr'].map(lambda x:item_nbr_transform(x,item_nbr_list))

X_all = pd.concat([X_train,X_val,X_test])
len_train = len(X_train)
len_val = len(X_val)
len_test = len(X_test)
del X_train,X_val,X_test
gc.collect()

cate_col = ['family','city','state','type']
for col in cate_col:
    dummies = pd.get_dummies(X_all[cate_col])
    dummies.columns = dummies.columns + col
    X_all =pd.concat([X_all,dummies],axis=1)
X_all = X_all.drop('item_nbr',axis=1)
X_all = X_all.drop('fc',axis=1)
X_all = X_all.drop(cate_col,axis=1)
X_all = X_all.fillna(0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_all)
X_all = scaler.transform(X_all)
print(X_all)
X_all = pd.DataFrame(X_all)

X_train = X_all.iloc[:len_train,:]
X_val = X_all.iloc[len_train:(len_train+len_val),:]
X_test = X_all.iloc[(len_train+len_val):,:]

del X_all

from keras.optimizers import Adam
optimizer = Adam(lr=0.001, decay=0.0)

reduce_lr_loss = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

val_pred = []
test_pred = []
# wtpath = 'weights.hdf5'  # To save best epoch. But need Keras bug to be fixed first.
#sample_weights=np.array( pd.concat([items["perishable"]] * 6) * 0.25 + 1 )
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    y = y_train[:, i]
    xv = np.array(X_val)
    
    
    yv = y_val[:, i]
    model = Sequential()

    model.add(Dense(512, input_dim=X_train.shape[1], init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(128, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(1, init='he_normal'))
    model.compile(loss = 'mse', optimizer=optimizer, metrics=['mse'])
    
    earlyStopping = EarlyStopping(monitor='mean_squared_error', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='mse', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    model.fit(np.array(X_train), y, batch_size=1024, epochs=100, verbose=1, 
              callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_data=(xv,yv))

    val_pred.append(model.predict(xv))
    test_pred.append(model.predict(np.array(X_test)))
    

    
n_public = 5 # Number of days in public test set
weights=pd.concat([items["perishable"]]) * 0.25 + 1
print("Unweighted validation mse: ", mean_squared_error(
    y_val, np.array(val_pred).squeeze(axis=2).transpose()) )

print("'Public' validation mse:   ", mean_squared_error(
    y_val[:,:n_public], np.array(val_pred).squeeze(axis=2).transpose()[:,:n_public], 
    sample_weight=weights) )
print("'Private' validation mse:  ", mean_squared_error(
    y_val[:,n_public:], np.array(val_pred).squeeze(axis=2).transpose()[:,n_public:], 
    sample_weight=weights) )


y_test = np.array(test_pred).transpose().reshape(167515,16)
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)




submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 10000)
submission.to_csv('NN_2-01-12.csv',float_format='%.5f',index=False)