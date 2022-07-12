import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

macro_cols = ["balance_trade", "average_provision_of_build_contract",
"micex_rgbi_tr", "micex_cbi_tr", "mortgage_value", "mortgage_rate", "cpi", "ppi",
"income_per_cap", "rent_price_4+room_bus", "apartment_build", "balance_trade_growth"]
# "deposits_rate",, , 

df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)
print(df_train.shape)
print(df_test.shape)

df_macro['ppi'] = df_macro['ppi']/315

df_macro['ppi'] = np.where(df_macro.ppi.isnull(), 1, df_macro.ppi)

# ========================
# Build df_all = (df_train+df_test).join(df_macro)
# ========================
df_train['env'] = 'train'
df_test['env'] = 'test'

id_test = df_test['id']

df_train.drop(['id'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)
df_test['price_doc'] = 0

num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')
print(df_all.shape)
print(id_test.shape)

# inflation adjustements
df_all['price_doc'] = df_all['price_doc'] / df_all['ppi']

ylog_all = np.log1p(df_all['price_doc'].values)

ylog_train_all = np.log1p(df_all[df_all['env']=='train']['price_doc'].values)

test_cpi_series = df_all['ppi'].values
test_cpi_series = df_all[df_all['env']=='test']['ppi'].values

df_all.drop(['price_doc'], axis=1, inplace=True)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)
df_all['month_year'] = month_year

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
df_all.drop(['timestamp'], axis=1, inplace=True)
df_all.drop(['market_shop_km'], axis=1, inplace=True)
df_all.drop(['green_part_5000'], axis=1, inplace=True)

# ========================
# Deal with categorical values
# ========================
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

# ========================
# Convert to numpy values
# ========================
X_all = df_values.values
print(X_all.shape)

# ========================
# Create a validation set, with last 20% of data
# ========================
num_val = int(num_train * 0.2)
num_val

X_train_all = X_all[:num_train]
X_train = X_all[:num_train-num_val]
X_val = X_all[num_train-num_val:num_train]
ylog_train = ylog_train_all[:-num_val]
ylog_val = ylog_train_all[-num_val:]

X_test = X_all[num_train:]

df_columns = df_values.columns

print('X_train_all shape is', X_train_all.shape)
print('X_train shape is', X_train.shape)
print('y_train shape is', ylog_train.shape)
print('X_val shape is', X_val.shape)
print('y_val shape is', ylog_val.shape)
print('X_test shape is', X_test.shape)

# error:
# providedpreds.size=24377, label.size=24376

dtrain_all = xgb.DMatrix(X_train_all, ylog_train_all, feature_names=df_columns)
dtrain = xgb.DMatrix(X_train, ylog_train, feature_names=df_columns)
dval = xgb.DMatrix(X_val, ylog_val, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

# ========================
# XGBOOST
# ========================
xgb_params = {
    'eta': 0.06,
    'max_depth': 5,
    'subsample': 1.0,
    'colsample_bytree': 0.6,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

# Uncomment to tune XGB `num_boost_rounds`
partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
                       early_stopping_rounds=20, verbose_eval=20)

# [200]	val-rmse:0.41588
# [205]	val-rmse:0.415762

num_boost_round = partial_model.best_iteration

#plt.figure(figsize=(100,20))
#fig, ax = plt.subplots(1, 1, figsize=(8, 16))
#xgb.plot_importance(partial_model, height=0.8, ax=ax)
#plt.show()

# ========================
# use best round for model
# ========================
model = xgb.train(dict(xgb_params, silent=1), dtrain_all, num_boost_round=num_boost_round)

#fig, ax = plt.subplots(1, 1, figsize=(8, 16))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

# ========================
# Submission
# ========================
ylog_pred = model.predict(dtest)
y_pred = (np.exp(ylog_pred) - 1) * test_cpi_series

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
df_sub.to_csv('sub.csv', index=False)

###################
