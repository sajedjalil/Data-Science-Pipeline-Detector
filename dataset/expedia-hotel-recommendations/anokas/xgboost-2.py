# Experimental script - does not produce prediction file yet

import numpy as np
import pandas as pd
import xgboost as xgb
import sys

trainloc = "../input/train.csv"
testloc = "../input/test.csv"
ssloc = "../input/sample_submission.csv"
train_cols = ['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country', 'hotel_cluster']

params = {}
params['objective'] = 'multi:softprob'
params['eval_metric'] = 'map@5'
params['num_class'] = 100

df_train = pd.DataFrame(columns=train_cols)
train_chunk = pd.read_csv(trainloc, chunksize=100000)
i = 0
for chunk in train_chunk:
    df_train = pd.concat([df_train, chunk[chunk['is_booking']==1][train_cols]])
    i = i + 1
    if i % 10 == 0:
        print("Rows loaded: " + str(i/10) + "mn")

print(df_train.head())
#print(df_train.shape())
x_train = df_train.drop(['hotel_cluster'])
y_train = df_train['hotel_cluster'].values

# Create train datamatrix
d_train = xgb.DMatrix(x_train, label=y_train)

clf = xgb.cv(params, d_train, num_boost_round=100000, early_stopping_rounds=50, verbose_eval=True, metrics='map@5')