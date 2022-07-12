import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

df_train.head()

# =============================
# =============================
# cleanup
# brings error down a lot by removing extreme price per sqm
print(df_train.shape)
df_train.loc[df_train.full_sq == 0, 'full_sq'] = 30
df_train = df_train[df_train.price_doc/df_train.full_sq <= 500000]
df_train = df_train[df_train.price_doc/df_train.full_sq >= 20000]
print(df_train.shape)
# =============================
# =============================

# ==============================
# for trend adjustment
df_train['month_year'] = (df_train.timestamp.dt.month + df_train.timestamp.dt.year * 100)

#df_train[df_train['month_year_cnt'] == 201206]
df_train.loc[df_train.month_year == 201207, 'price_doc'] = df_train['price_doc'] / 0.93
df_train.loc[df_train.month_year == 201208, 'price_doc'] = df_train['price_doc'] / 0.93
df_train.loc[df_train.month_year == 201209, 'price_doc'] = df_train['price_doc'] / 0.92
df_train.loc[df_train.month_year == 201210, 'price_doc'] = df_train['price_doc'] / 0.92
df_train.loc[df_train.month_year == 201211, 'price_doc'] = df_train['price_doc'] / 0.91
df_train.loc[df_train.month_year == 201212, 'price_doc'] = df_train['price_doc'] / 0.90

df_train.loc[df_train.month_year == 201301, 'price_doc'] = df_train['price_doc'] / 0.90
df_train.loc[df_train.month_year == 201302, 'price_doc'] = df_train['price_doc'] / 0.91
df_train.loc[df_train.month_year == 201303, 'price_doc'] = df_train['price_doc'] / 0.92
df_train.loc[df_train.month_year == 201304, 'price_doc'] = df_train['price_doc'] / 0.92
df_train.loc[df_train.month_year == 201305, 'price_doc'] = df_train['price_doc'] / 0.94
df_train.loc[df_train.month_year == 201306, 'price_doc'] = df_train['price_doc'] / 0.95
df_train.loc[df_train.month_year == 201307, 'price_doc'] = df_train['price_doc'] / 0.97
df_train.loc[df_train.month_year == 201308, 'price_doc'] = df_train['price_doc'] / 0.98
df_train.loc[df_train.month_year == 201309, 'price_doc'] = df_train['price_doc'] / 0.99
df_train.loc[df_train.month_year == 201310, 'price_doc'] = df_train['price_doc'] / 1.00
df_train.loc[df_train.month_year == 201311, 'price_doc'] = df_train['price_doc'] / 1.01
df_train.loc[df_train.month_year == 201312, 'price_doc'] = df_train['price_doc'] / 1.03

df_train.loc[df_train.month_year == 201401, 'price_doc'] = df_train['price_doc'] / 1.03
df_train.loc[df_train.month_year == 201402, 'price_doc'] = df_train['price_doc'] / 1.04
df_train.loc[df_train.month_year == 201403, 'price_doc'] = df_train['price_doc'] / 1.05
df_train.loc[df_train.month_year == 201404, 'price_doc'] = df_train['price_doc'] / 1.06
df_train.loc[df_train.month_year == 201405, 'price_doc'] = df_train['price_doc'] / 1.07
df_train.loc[df_train.month_year == 201406, 'price_doc'] = df_train['price_doc'] / 1.07
df_train.loc[df_train.month_year == 201407, 'price_doc'] = df_train['price_doc'] / 1.09
df_train.loc[df_train.month_year == 201408, 'price_doc'] = df_train['price_doc'] / 1.10
df_train.loc[df_train.month_year == 201409, 'price_doc'] = df_train['price_doc'] / 1.11
df_train.loc[df_train.month_year == 201410, 'price_doc'] = df_train['price_doc'] / 1.12
df_train.loc[df_train.month_year == 201411, 'price_doc'] = df_train['price_doc'] / 1.13
df_train.loc[df_train.month_year == 201412, 'price_doc'] = df_train['price_doc'] / 1.14


# ==============================
y_train = df_train['price_doc'].values
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

# Build df_all = (df_train+df_test).join(df_macro)
num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)


# ==============================
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

# ==============================
# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)

factorize = lambda t: pd.factorize(t[1])[0]

df_obj = df_all.select_dtypes(include=['object'])

X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]


# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

df_columns = df_values.columns


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


# Uncomment to tune XGB `num_boost_rounds`

#cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#    verbose_eval=True, show_stdv=False)
#cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()
#num_boost_rounds = len(cv_result)

num_boost_round = 489

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)

fig, ax = plt.subplots(1, 1, figsize=(8, 16))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

y_pred = model.predict(dtest)

###############
# adjsuting for latest known trend ratio
y_pred = y_pred * 1.14


df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

df_sub.to_csv('sub.csv', index=False)

