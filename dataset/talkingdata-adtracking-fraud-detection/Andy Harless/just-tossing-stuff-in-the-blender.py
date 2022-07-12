# References:
# https://www.kaggle.com/shujian/mlp-starter
# https://www.kaggle.com/titericz/giba-darragh-ftrl-rerevisited
# https://www.kaggle.com/joaopmpeinado/talkingdata-xgboost-lb-0-951
# https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data
# https://www.kaggle.com/cartographic/undersampler
# https://www.kaggle.com/prashantkikani/weighted-app-chanel-os
# https://www.kaggle.com/ogrellier/ftrl-in-chunck

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.special import expit, logit

almost_zero = 1e-10
almost_one = 1 - almost_zero

models = {
  'xgb  ':  "../input/xgboost-lb-0-951/xgb_sub.csv",
  'ftrl1':  "../input/giba-darragh-ftrl-rerevisited/sub_proba.csv",
  'nn   ':  "../input/shujian-s-mlp-starter-9502-version/sub_mlp.csv",
  'lgb  ':  "../input/pranav-s-lightgbm-9631-version/sub_lgb_balanced99.csv",
  'usam ':  "../input/undersampler/pred.csv",
  'means':  "../input/weighted-app-chanel-os/subnew.csv",
  'ftrl2':  "../input/ftrl-in-chunck/ftrl_submission.csv"
  }
  
weights = {
  'xgb  ':  .02,
  'ftrl1':  .09,
  'nn   ':  .04,
  'lgb  ':  .60,
  'usam ':  .05,
  'means':  .10,
  'ftrl2':  .10
  }
  
print (sum(weights.values()))

subs = {m:pd.read_csv(models[m]) for m in models}

logits = {s:subs[s]['is_attributed'].clip(almost_zero,almost_one).apply(logit) for s in subs}

weighted = 0
for m in logits:
    s = logits[m].std()
    print(m, s)
    weighted = weighted + weights[m]*logits[m] / s


final_sub = pd.DataFrame()
final_sub['click_id'] = subs['xgb  ']['click_id']
final_sub['is_attributed'] = weighted.apply(expit)

final_sub.to_csv("sub_mix.csv", index=False)