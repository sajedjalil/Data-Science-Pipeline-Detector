# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
color = sns.color_palette()


pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)

train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
test_df = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
macro_df = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])
train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')
test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')
print(train_df.shape, test_df.shape)

# truncate the extreme values in price_doc #
ulimit = np.percentile(train_df.price_doc.values, 99)
llimit = np.percentile(train_df.price_doc.values, 1)
train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit


for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values.astype('str')) + list(test_df[f].values.astype('str')))
        train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))
        test_df[f] = lbl.transform(list(test_df[f].values.astype('str')))
        
train_df.fillna(-99, inplace=True)
test_df.fillna(-99, inplace=True)

train_df["yearmonth"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.month
test_df["yearmonth"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.month

# year and week #
train_df["yearweek"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.weekofyear
test_df["yearweek"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.weekofyear

# year #
train_df["year"] = train_df["timestamp"].dt.year
test_df["year"] = test_df["timestamp"].dt.year

# month of year #
train_df["month_of_year"] = train_df["timestamp"].dt.month
test_df["month_of_year"] = test_df["timestamp"].dt.month

# week of year #
train_df["week_of_year"] = train_df["timestamp"].dt.weekofyear
test_df["week_of_year"] = test_df["timestamp"].dt.weekofyear

# day of week #
train_df["day_of_week"] = train_df["timestamp"].dt.weekday
test_df["day_of_week"] = test_df["timestamp"].dt.weekday


train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
test_X = test_df.drop(["id", "timestamp"] , axis=1)

train_y = np.log1p(train_df.price_doc.values)

train_X.fillna(0, inplace=True)
train_X.replace([np.inf, -np.inf], np.nan)
train_X = train_X[train_X < 1.7976931348623157e+308]

val_time = 201407
dev_indices = np.where(train_X["yearmonth"]<val_time)
val_indices = np.where(train_X["yearmonth"]>=val_time)
dev_X = train_X.ix[dev_indices]
val_X = train_X.ix[val_indices]
dev_y = train_y[dev_indices]
val_y = train_y[val_indices]

features_to_use = ["full_sq", "floor", "railroad_km", "green_zone_km", "brent", "state", "mosque_km", 
                   "micex_cbi_tr", "bus_terminal_avto_km", "big_road1_km", "metro_km_avto", "life_sq",
                   "public_healthcare_km", "workplaces_km", "indust_part", "kindergarten_km", 
                   "public_transport_station_km", "theater_km", "park_km", "oil_chemistry_km", 
                  "cafe_count_2000", "cafe_count_5000", "additional_education_km", 
                   "power_transmission_line_km", "incineration_km", "green_part_1500", "sub_area", "metro_min_avto", "thermal_power_plant_km", 
                  "industrial_km","green_part_5000", "railroad_station_avto_km", "cemetery_km", 
                  "church_synagogue_km", "area_m","trc_sqm_3000", "big_market_km", 
                  "metro_min_walk"]
                  
                  
                  
dev_X_new = dev_X
val_X_new = val_X

xgb_params = {
    'eta': 0.05,
    'max_depth': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'min_child_weight':1,
    'silent': 1,
    'seed':0
}


xgtrain = xgb.DMatrix(dev_X_new, dev_y, feature_names=dev_X_new.columns)
xgtest = xgb.DMatrix(val_X_new, val_y, feature_names=val_X_new.columns)
watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
num_rounds = 200 # Increase the number of rounds while running in local
model = xgb.train(xgb_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=5)


test_X = test_X
test = xgb.DMatrix(test_X)
pred = model.predict(test)

sub = pd.DataFrame()
sub['id'] = test_df['id'].values
sub['price_doc'] = pred
sub.to_csv('xgb.csv', index=False)



















