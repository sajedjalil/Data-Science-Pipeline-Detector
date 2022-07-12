import numpy as np
import pandas as pd
import xgboost as xgb



df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
#df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

print(len(df_train))
df_train['price_doc'].dropna(inplace=True)


feature_list=['full_sq','life_sq','floor','max_floor','build_year']

df_train[feature_list].dropna(inplace=True)

print(len(df_train))













#model = xgb.train(dict(xgb_params, silent=0), dtrain_all, num_boost_round=num_boost_round)

#ylog_pred = model.predict(dtest)
#y_pred = np.exp(ylog_pred) - 1

#df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

#df_sub.to_csv('sub0205201701.csv', index=False)



