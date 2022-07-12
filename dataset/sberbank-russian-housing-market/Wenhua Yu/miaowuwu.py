# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime

#load files
train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])
id_test = test.id

#multiplier = 0.969
# truncate the extreme values in price_doc #
Q1 = np.percentile(np.log1p(train.price_doc.values), 25)
Q2 = np.percentile(np.log1p(train.price_doc.values), 50)
Q3 = np.percentile(np.log1p(train.price_doc.values), 75)
IQR=Q3 - Q1
infbdd=Q1 - 1.5 * IQR 
supbdd=Q3 + 1.5 * IQR 
ibdd=int(np.exp(infbdd))

sbdd=int( np.exp(supbdd))
sindex=train[train.price_doc > sbdd].index
iindex=train[train.price_doc < ibdd].index
train.loc[iindex,"price_doc"] = ibdd
train.loc[sindex,"price_doc"] = sbdd

#clean data
#test with dot error
equal_index = [601,1896,2791]
test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]
#train,test full life with 0,1
bad_index = train[train.life_sq < 2].index
train.loc[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq < 2].index
test.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 2].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq < 2].index
test.loc[bad_index, "full_sq"] = np.NaN

#test full<life 
exchange_index=[64,119,171]
life_bad_index=[2027, 2031, 5187]
full_bad_index=[2804]
test.loc[life_bad_index, "life_sq"] = np.NaN
test.loc[full_bad_index, "full_sq"] = np.NaN
for cat in exchange_index:
    dog=test.loc[cat, "life_sq"]
    test.loc[cat, "life_sq"] = test.loc[cat, "full_sq"]
    test.loc[cat, "full_sq"]=dog 
    #train full<life

life_bad_index=train[(train.life_sq > train.full_sq) & ((np.log1p(train.price_doc)/np.log1p(train.full_sq)) > 3)&((np.log1p(train.price_doc)/np.log1p(train.full_sq)) < 5.25)].index
train.loc[life_bad_index, "life_sq"] = np.NaN
full_bad_index=train[(train.life_sq > train.full_sq)&((np.log1p(train.price_doc)/np.log1p(train.life_sq)) > 3.13)&((np.log1p(train.price_doc)/np.log1p(train.life_sq)) < 6.31)].index
train.loc[full_bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > train.full_sq].index
train.loc[bad_index, "life_sq"] = np.NaN

#train life/full too small version 6 use 0.3 best

life_bad_index=train[(train.life_sq / train.full_sq < 0.3) & ((np.log1p(train.price_doc)/np.log1p(train.full_sq)) > 3)&((np.log1p(train.price_doc)/np.log1p(train.full_sq)) < 5.25)].index
train.loc[life_bad_index, "life_sq"] = np.NaN
full_bad_index=train[(train.life_sq / train.full_sq < 0.3)&((np.log1p(train.price_doc)/np.log1p(train.life_sq)) > 3.13)&((np.log1p(train.price_doc)/np.log1p(train.life_sq)) < 6.31)].index
train.loc[full_bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq / train.full_sq < 0.3].index
train.loc[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq / test.full_sq < 0.3].index
test.loc[bad_index, "life_sq"] = np.NaN

#focus on kitchen

kitch_is_build_year = [13117]
train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.loc[bad_index, "kitch_sq"] = np.NaN



k=np.nanpercentile(train.kitch_sq/train.full_sq, 99.9)
bad_index = train[train.kitch_sq/train.full_sq > k].index
train.loc[bad_index, "kitch_sq"] = np.NaN

bad_index = test[test.kitch_sq/test.full_sq > k].index
test.loc[bad_index, "kitch_sq"] = np.NaN


bad_index=train[train.life_sq + train.kitch_sq > train.full_sq].index
train.loc[bad_index,"life_sq"]=train.loc[bad_index,"full_sq"]-train.loc[bad_index,"kitch_sq"]
bad_index=test[test.life_sq + test.kitch_sq > test.full_sq].index
test.loc[bad_index,"life_sq"]=test.loc[bad_index,"full_sq"]-test.loc[bad_index,"kitch_sq"]



#typevalue
train.product_type.value_counts(normalize= True)
test.product_type.value_counts(normalize= True)

#buildyear
bad_index = train[train.build_year < 1500].index
train.loc[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year < 1500].index
test.loc[bad_index, "build_year"] = np.NaN
bad_index = train[train.build_year >2020].index
train.loc[bad_index, "build_year"] = np.NaN

#num_room
bad_index = train[train.num_room == 0].index 
train.loc[bad_index, "num_room"] = np.NaN
bad_index = test[test.num_room == 0].index 
test.loc[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.loc[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
test.loc[bad_index, "num_room"] = np.NaN

#floor
bad_index = train[train.max_floor > 60].index
train.loc[bad_index, "max_floor"] = np.NaN

bad_index = train[train.floor == 0].index
train.loc[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.loc[bad_index, "max_floor"] = np.NaN

bad_index = test[test.max_floor == 0].index
test.loc[bad_index, "max_floor"] = np.NaN

bad_index = train[train.floor > train.max_floor].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.floor > test.max_floor].index
test.loc[bad_index, "max_floor"] = np.NaN
train.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train.loc[bad_index, "floor"] = np.NaN

#material
train.material.value_counts()
test.material.value_counts()

#state
train.state.value_counts()
bad_index = train[train.state == 33].index
train.loc[bad_index, "state"] = np.NaN
test.state.value_counts()

#extreme price
train=train[((np.log1p(train.price_doc)/np.log1p(train.life_sq))>3.13)]
train=train[((np.log1p(train.price_doc)/np.log1p(train.life_sq)) <6.31)]
#train=train[(train.price_doc)/(train.full_sq) > 10000]

# Add month-year
month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
train['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
train['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
train['month'] = train.timestamp.dt.month
train['dow'] = train.timestamp.dt.dayofweek

test['month'] = test.timestamp.dt.month
test['dow'] = test.timestamp.dt.dayofweek

# Other feature engineering
train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)
train['rel_living'] = train['life_sq'] / train['full_sq'].astype(float)
train['rel_kitch_life'] = train['kitch_sq'] / train['life_sq'].astype(float)

test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)
test['rel_living'] = test['life_sq'] / test['full_sq'].astype(float)
test['rel_kitch_life'] = test['kitch_sq'] / test['life_sq'].astype(float)

#train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
#test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
test['room_size'] = test['life_sq'] / test['num_room'].astype(float)
train["yearmonth"] = train["timestamp"].dt.year*100 + train["timestamp"].dt.month
test["yearmonth"] = test["timestamp"].dt.year*100 + test["timestamp"].dt.month


y_train = np.log1p(train.price_doc.values)
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)




for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)
        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        #x_test.drop(c,axis=1,inplace=True)  


xgb_params = {
    'eta': 0.05,
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'min_child_weight':1,
    'silent': 1,
    'seed':0
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

#cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#    verbose_eval=50, show_stdv=False)
#cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

#num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=261)

#fig, ax = plt.subplots(1, 1, figsize=(8, 13))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

y_predict = model.predict(dtest)
y_predict = np.round(np.expm1(y_predict))
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()

output.to_csv('try_miao2.csv', index=False)





















from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.