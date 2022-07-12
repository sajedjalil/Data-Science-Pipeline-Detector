# This is based on the content from: 
# 1. https://www.kaggle.com/dott1718/922-in-3-minutes by @dott1718
# 2. https://www.kaggle.com/titericz/giba-single-model-public-0-9245-private-0-9234
# 3. https://www.kaggle.com/nawidsayed/lightgbm-and-cnn-3rd-place-solution
# This is only a mod, I'm trying new things and to improve the original in time/result.

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import logit

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
features = [x for x in train_df.columns if x.startswith("var")]

"""
from Giba
#Reverse features
for var in features:
    if np.corrcoef( train_df['target'], train_df[var] )[1][0] < 0:
        train_df[var] = train_df[var] * -1
        test_df[var]  = test_df[var]  * -1
        
#count all values
var_stats = {}
hist_df = pd.DataFrame()
for var in features:
    var_stats = train_df[var].append(test_df[var]).value_counts()
    hist_df[var] = pd.Series(test_df[var]).map(var_stats)
    hist_df[var] = hist_df[var] > 1
#remove fake test rows
ind = hist_df.sum(axis=1) != 200

"""

for var in features:
    if np.corrcoef( train_df['target'], train_df[var] )[1][0] < 0:
        train_df[var] = train_df[var] * -1
        test_df[var]  = test_df[var]  * -1
        
hist_df = pd.DataFrame()
for var in features:
    var_stats = train_df[var].append(test_df[var]).value_counts()
    hist_df[var] = pd.Series(test_df[var]).map(var_stats)
    hist_df[var] = hist_df[var] > 1

ind = hist_df.sum(axis=1) != 200
var_stats = {var:train_df[var].append(test_df[ind][var]).value_counts() for var in features}

pred = 0
for var in features:

    model = lgb.LGBMClassifier(**{ 'learning_rate':0.06, 'max_bin': 165, 'max_depth': 5, 'min_child_samples': 153,
        'min_child_weight': 0.1, 'min_split_gain': 0.0018, 'n_estimators': 41, 'num_leaves': 6, 'reg_alpha': 2.1,
        'reg_lambda': 2.54, 'objective': 'binary', 'n_jobs': -1})
        
    model = model.fit(np.hstack([train_df[var].values.reshape(-1,1),
                                 train_df[var].map(var_stats[var]).values.reshape(-1,1)]),
                               train_df["target"].values)
    pred += logit(model.predict_proba(np.hstack([test_df[var].values.reshape(-1,1),
                                 test_df[var].map(var_stats[var]).values.reshape(-1,1)]))[:,1])
    
pd.DataFrame({"ID_code":test_df["ID_code"], "target":pred}).to_csv("submission.csv", index=False)

