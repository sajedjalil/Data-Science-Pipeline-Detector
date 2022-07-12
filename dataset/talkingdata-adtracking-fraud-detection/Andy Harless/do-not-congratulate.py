
LOGIT_WEIGHT = .8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.special import expit, logit

almost_zero = 1e-10
almost_one = 1 - almost_zero

models = {
  'xgb  ':  "../input/jo-o-s-xgboost-with-memory-usage-enhancements/xgb_sub.csv",
  'ftrl1':  "../input/giba-darragh-ftrl-rerevisited/sub_proba.csv",
  'nn   ':  "../input/shujian-s-mlp-starter-9502-version/sub_mlp.csv",
  'lgb  ':  "../input/nooh-s-lgbm-smaller/sub_lgb_balanced55.csv",
  'usam ':  "../input/lewis-undersampler-9562-version/pred.csv",
  'means':  "../input/weighted-app-chanel-os/subnew.csv",
  'ftrl2':  "../input/olivier-s-multi-process-ftrl-9619-version/ftrl_submission.csv",
  'nn2  ':  "../input/alexander-kireev-s-imbalanced-dl/imbalanced_data.csv",
  'rlgb ':  "../input/pranav-s-r-lightgbm-9683-version/sub_lightgbm_R_reduced.csv",
  'nn2a ':  "../input/alexander-kireev-s-dl/dl_support.csv",
  'xgb2 ':  "../input/swetha-s-xgboost-revised/xgb_sub5.csv"
  }
  
weights = {
  'xgb  ':  .02,
  'ftrl1':  .01,
  'nn   ':  .02,
  'lgb  ':  .17,
  'usam ':  .03,
  'means':  .03,
  'ftrl2':  .05,
  'nn2  ':  .20,
  'rlgb ':  .40,
  'nn2a ':  .05,
  'xgb2 ':  .02
  }
  
print (sum(weights.values()))


subs = {m:pd.read_csv(models[m]) for m in models}
first_model = list(models.keys())[0]
n = subs[first_model].shape[0]

ranks = {s:subs[s]['is_attributed'].rank()/n for s in subs}
logits = {s:subs[s]['is_attributed'].clip(almost_zero,almost_one).apply(logit) for s in subs}

logit_avg = 0
rank_avg = 0
for m in models:
    s = logits[m].std()
    print(m, s)
    logit_avg = logit_avg + weights[m]*logits[m] / s
    rank_avg = rank_avg + weights[m]*ranks[m]

logit_rank_avg = logit_avg.rank()/n
final_avg = LOGIT_WEIGHT*logit_rank_avg + (1-LOGIT_WEIGHT)*rank_avg

final_sub = pd.DataFrame()
final_sub['click_id'] = subs[first_model]['click_id']
final_sub['is_attributed'] = final_avg

print( final_sub.head() )

final_sub.to_csv("sub_mix_logits_ranks.csv", index=False, float_format='%.8f')