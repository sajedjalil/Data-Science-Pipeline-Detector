__author__ = 'MT'

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test['loss'] = np.nan
merged = pd.concat([train, test])

num_features = [n for n in merged.columns if n.startswith('cont')]
cat_features = [n for n in merged.columns if n.startswith('cat')]


print (len(cat_features))
for cat in cat_features:
    temp_series = train[cat]
    levels = len(temp_series.unique())
    if levels == 2:
        print( cat, levels)





