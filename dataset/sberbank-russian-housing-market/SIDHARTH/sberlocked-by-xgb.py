
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
#multiplier = 0.969
mult = .969

train.drop(train[train["life_sq"] > 7000].index, inplace=True)
y_train = train['price_doc']#.values  * mult + 10

id_test = test['id']

train.drop(['id', 'price_doc'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

num_train = len(train)
df_all = pd.concat([train, test])
# Next line just adds a lot of NA columns (becuase "join" only works on indexes)
# but somewhow it seems to affect the result
df_all = df_all.join(macro, on='timestamp', rsuffix='_macro')

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
#df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)

#clean data
bad_index = df_all[df_all.life_sq > df_all.full_sq].index
df_all.ix[bad_index, "life_sq"] = np.NaN
#equal_index = [601,1896,2791]
#test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]
#bad_index = test[test.life_sq > test.full_sq].index
#test.ix[bad_index, "life_sq"] = np.NaN
bad_index = df_all[df_all.life_sq < 5].index
df_all.ix[bad_index, "life_sq"] = np.NaN
#bad_index = test[test.life_sq < 5].index
#test.ix[bad_index, "life_sq"] = np.NaN
bad_index = df_all[df_all.full_sq < 5].index
df_all.ix[bad_index, "full_sq"] = np.NaN
#bad_index = test[test.full_sq < 5].index
#test.ix[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
df_all.ix[kitch_is_build_year, "build_year"] = df_all.ix[kitch_is_build_year, "kitch_sq"]
bad_index = df_all[df_all.kitch_sq >= df_all.life_sq].index
df_all.ix[bad_index, "kitch_sq"] = np.NaN
#bad_index = test[test.kitch_sq >= test.life_sq].index
#test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index =df_all[(df_all.kitch_sq == 0).values + (df_all.kitch_sq == 1).values].index
df_all.ix[bad_index, "kitch_sq"] = np.NaN
#bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
#test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = df_all[(df_all.full_sq > 210) & (df_all.life_sq / df_all.full_sq < 0.3)].index
df_all.ix[bad_index, "full_sq"] = np.NaN
#bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
#test.ix[bad_index, "full_sq"] = np.NaN
bad_index = df_all[df_all.life_sq > 300].index
df_all.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
#bad_index = test[test.life_sq > 200].index
#test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
df_all.product_type.value_counts(normalize= True)
#test.product_type.value_counts(normalize= True)
bad_index = df_all[df_all.build_year < 1500].index
df_all.ix[bad_index, "build_year"] = np.NaN
#bad_index = test[test.build_year < 1500].index
#test.ix[bad_index, "build_year"] = np.NaN
bad_index = df_all[df_all.num_room == 0].index 
df_all.ix[bad_index, "num_room"] = np.NaN
#bad_index = test[test.num_room == 0].index 
#test.ix[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
df_all.ix[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
#test.ix[bad_index, "num_room"] = np.NaN
bad_index = df_all[(df_all.floor == 0).values * (df_all.max_floor == 0).values].index
df_all.ix[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = df_all[df_all.floor == 0].index
df_all.ix[bad_index, "floor"] = np.NaN
bad_index = df_all[df_all.max_floor == 0].index
df_all.ix[bad_index, "max_floor"] = np.NaN
#bad_index = test[test.max_floor == 0].index
#test.ix[bad_index, "max_floor"] = np.NaN
bad_index = df_all[df_all.floor > df_all.max_floor].index
df_all.ix[bad_index, "max_floor"] = np.NaN
#bad_index = test[test.floor > test.max_floor].index
#test.ix[bad_index, "max_floor"] = np.NaN
df_all.floor.describe(percentiles= [0.9999])
bad_index = [23584]
df_all.ix[bad_index, "floor"] = np.NaN
df_all.material.value_counts()
#test.material.value_counts()
df_all.state.value_counts()
bad_index = df_all[df_all.state == 33].index
df_all.ix[bad_index, "state"] = np.NaN
#test.state.value_counts()

# Other feature engineering
df_all.apartment_name=df_all.sub_area + df_all['metro_km_avto'].astype(str)
#test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

df_all['room_size'] = df_all['life_sq'] / df_all['num_room'].astype(float)
#test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

df_all.loc[df_all.full_sq == 0, 'full_sq'] = 50

X_train = df_all[:num_train]
X_test = df_all[num_train:]
del df_all, train, test, macro
# brings error down a lot by removing extreme price per sqm

X_train['price_doc'] =  y_train

#print(X_train.head())

X_train = X_train[X_train.price_doc/X_train.full_sq <= 600000]
X_train = X_train[X_train.price_doc/X_train.full_sq >= 10000]
y_train = X_train["price_doc"]



x_train = X_train.drop(["timestamp", "price_doc", "timestamp_macro"], axis=1)
x_test = X_test.drop(["timestamp", "timestamp_macro"], axis=1)

del X_train, X_test

num_train = len(x_train)
x_all = pd.concat([x_train, x_test])
del x_train, x_test
#print(x_all.head())
#print(x_all.shape)

factorize = lambda t: pd.factorize(t[1])[0]

df_obj = x_all.select_dtypes(include=['object'])

X_all = np.c_[
    x_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]
print(X_all.shape)

# Deal with categorical values
df_numeric = x_all.select_dtypes(exclude=['object'])
df_obj = x_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

# Convert to numpy values
X_all = df_values.values
'''
#x_all = x_all.fillna(0)
for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values)) 
        x_all[c] = lbl.transform(list(x_all[c].values))
        #x_train.drop(c,axis=1,inplace=True)
print(x_all.shape)
'''
X_train = X_all[:num_train]
X_test = X_all[num_train:]

#X_train = X_train.astype(float)
#X_test = X_test.astype(float)
df_columns = df_values.columns


del x_all, X_all

#print(x_train.shape)
#print(x_train.head())

#dtrain = xgb.DMatrix(x_train, y_train)
#dtest = xgb.DMatrix(x_test)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=10000, early_stopping_rounds=50,
    verbose_eval=50, show_stdv=False)
#cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_rounds)

#fig, ax = plt.subplots(1, 1, figsize=(8, 13))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

y_predict = model.predict(dtest)
y_predict = np.round(y_predict * 0.99)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()
output.to_csv('sub.csv', index=False)
