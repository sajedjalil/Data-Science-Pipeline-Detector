import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from scipy import stats
import random

df_train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
df_test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

id_test = df_test['id']
y_train = df_train['price_doc'] * .95 + 10.05

num_train = len(df_train)
df_all = pd.concat([df_train, df_test])

# Missing values build_year..
df_all.loc[np.absolute(df_all['build_year'] - df_all.timestamp.dt.year) > 150, "build_year"] = df_all.timestamp.dt.year.values
build_year_per_area = df_all[["build_year", "sub_area"]].groupby("sub_area").aggregate(lambda x: stats.mode(x).mode[0]).to_dict()["build_year"]
df_all.loc[df_all.build_year.isnull(), "build_year"] = df_all.sub_area.map(lambda x: build_year_per_area[x])

# Wrong values life_sq
life_sq_mode = stats.mode(df_all['life_sq']).mode[0]
df_all['life_sq'].fillna(life_sq_mode, inplace=True)
df_all.loc[df_all['life_sq'] > df_all['full_sq'], 'life_sq'] = df_all['full_sq'].values

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
# df_all['month_year'] = df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100
# df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

df_all['floor_from_top'] = df_all['max_floor'] - df_all['floor']
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor']
df_all['avg_room_size'] = (df_all['life_sq'] - df_all['kitch_sq']) / df_all['num_room']
df_all['prop_living'] = df_all['life_sq'] / df_all['full_sq']
df_all['prop_kitchen'] = df_all['kitch_sq'] / df_all['full_sq']
df_all['extra_area'] = df_all['full_sq'] - df_all['life_sq']
df_all['age_at_sale'] = df_all['build_year'] - df_all.timestamp.dt.year

df_all['ratio_preschool'] = df_all['children_preschool'] / df_all['preschool_quota']
df_all['ratio_school'] = df_all['children_school'] / df_all['school_quota']

## Appartment building sales per month feature
building_year_month = df_all['sub_area'] +\
                      df_all['metro_km_avto'].astype(str) +\
                      (df_all.timestamp.dt.month + \
                       df_all.timestamp.dt.year * 100).astype(str)
building_year_month_cnt_map = building_year_month.value_counts().to_dict()
df_all['building_year_month_cnt'] = building_year_month.map(building_year_month_cnt_map)
        
df_all.drop(['id', 'timestamp', 'price_doc'], axis=1, inplace=True)

df_all = pd.get_dummies(df_all, drop_first=True)

x_train = df_all[:num_train]
x_test = df_all[num_train:]

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

#
# Train N slightly different models and average their outcome
#

predictions = None
num_models = 3

for i in range(1,num_models+1):
    xgb_params.update(colsample_bytree=0.7 + random.uniform(-0.15, 0.15),
                      subsample=0.7 + random.uniform(-0.15, 0.15),
                      max_depth=5 - random.randint(0,1))
                      
    m = xgb.train(xgb_params, dtrain, num_boost_round=400 + random.randint(-35,25))
        
    if predictions == None:
        predictions = m.predict(dtest)
    else:
        predictions = predictions + m.predict(dtest)

    print ('Trained XGB model %d' % i)

output = pd.DataFrame({'id': id_test, 'price_doc': predictions / float(num_models)})

output.to_csv('submission.csv', index=False)
