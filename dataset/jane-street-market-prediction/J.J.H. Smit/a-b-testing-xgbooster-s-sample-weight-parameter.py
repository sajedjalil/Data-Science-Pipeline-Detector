import numpy as np
import pandas as pd
import xgboost as xgb

import janestreet


dtrain = pd.read_parquet('../input/dtrain-parquet/dtrain.parquet')
dlabels = dtrain[['date', 'weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']]
dtrain = dtrain.drop(dlabels.columns, axis=1)
dlabels['action'] = (dlabels['resp'] > 0).astype('int')

booster = xgb.XGBClassifier(
    tree_method='gpu_hist',
    n_jobs=-1,
)
booster.fit(
    X=dtrain,
    y=dlabels['action'],
    sample_weight=dlabels['weight'],
)

env = janestreet.make_env()
for (dtest, dpredict) in env.iter_test():
    dpredict['action'] = booster.predict(dtest[[c for c in dtest.columns if 'feature_' in c]])
    env.predict(dpredict)