# coding=utf8
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

RS = 20170501
np.random.seed(RS)

ROUNDS = 450
params = {
	'objective': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'learning_rate': 0.04,
    'verbose': 0,
    'num_leaves': 2 ** 5,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': RS,
    'feature_fraction': 0.7,
    'feature_fraction_seed': RS,
    'max_bin': 100,
    'max_depth': 5,
    'num_rounds': ROUNDS
}

print("Started")
input_folder = '../input/'
train_df = pd.read_csv(input_folder + 'train.csv', parse_dates=['timestamp'])
test_df  = pd.read_csv(input_folder + 'test.csv' , parse_dates=['timestamp'])
macro_df = pd.read_csv(input_folder + 'macro.csv', parse_dates=['timestamp'])

#fix outlier
train_df.drop(train_df[train_df["life_sq"] > 5000].index, inplace=True)

train_y  = np.log(train_df['price_doc'].values)
test_ids = test_df['id']

train_df.drop(['id', 'price_doc'], axis=1, inplace=True)
test_df.drop(['id'], axis=1, inplace=True)
print("Data: X_train: {}, X_test: {}".format(train_df.shape, test_df.shape))

df = pd.concat([train_df, test_df])

#Lets try using only those from https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity
macro_cols = ["timestamp","balance_trade","balance_trade_growth","eurrub","average_provision_of_build_contract","micex_rgbi_tr","micex_cbi_tr","deposits_rate","mortgage_value","mortgage_rate","income_per_cap","museum_visitis_per_100_cap","apartment_build"]
df = df.merge(macro_df[macro_cols], on='timestamp', how='left')
print("Merged with macro: {}".format(df.shape))

#Dates...
df['year'] = df.timestamp.dt.year
df['month'] = df.timestamp.dt.month
df['dow'] = df.timestamp.dt.dayofweek
df.drop(['timestamp'], axis=1, inplace=True)

#More featuers needed...

df_num = df.select_dtypes(exclude=['object'])
df_obj = df.select_dtypes(include=['object']).copy()
for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_num, df_obj], axis=1)

pos = train_df.shape[0]
train_df = df_values[:pos]
test_df  = df_values[pos:]
del df, df_num, df_obj, df_values

print("Training on: {}".format(train_df.shape, train_y.shape))

train_lgb = lgb.Dataset(train_df, train_y)
model = lgb.train(params, train_lgb, num_boost_round=ROUNDS)
preds = model.predict(test_df)
	
print("Writing output...")
out_df = pd.DataFrame({"id":test_ids, "price_doc":np.exp(preds)})
out_df.to_csv("lgb_{}_{}.csv".format(ROUNDS, RS), index=False)
print(out_df.head(3))

print("Features importance...")
gain = model.feature_importance('gain')
ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(ft.head(25))

plt.figure()
ft[['feature','gain']].head(25).plot(kind='barh', x='feature', y='gain', legend=False, figsize=(10, 20))
plt.gcf().savefig('features_importance.png')

print("Done.")