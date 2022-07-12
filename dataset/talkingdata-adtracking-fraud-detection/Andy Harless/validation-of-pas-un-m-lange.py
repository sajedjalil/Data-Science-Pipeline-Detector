
LOGIT_WEIGHT = 0.3
PUBLIC_CUTOFF = 4032690

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

almost_zero = 1e-10
almost_one = 1 - almost_zero

models = {
  'part1':  "../input/validate-krishna-s-r-lgbm-with-time-deltas-part-1/lgb_Usrnewness_val.csv",
  'part2':  "../input/validate-krishna-s-r-lgbm-with-time-deltas-part-2/lgb_Usrnewness_val.csv",
  'part3':  "../input/validate-krishna-s-r-lgbm-with-time-deltas-part-3/lgb_Usrnewness_val.csv",
  }
 
weights = {
  'part1':  .53,
  'part2':  .27,
  'part3':  .14,
  }
  
tot = sum(weights.values())
for w in weights:
    weights[w] /= tot

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
final_sub['click_id'] = range(n)
final_sub['is_attributed'] = final_avg

print( final_sub.head() )

y_test = pd.read_pickle('../input/training-and-validation-data-pickle/validation.pkl.gz')['is_attributed'].values
y_pred = final_sub['is_attributed'].values
print(  "\n\nFULL VALIDATION SCORE:    ", 
        roc_auc_score( y_test, y_pred )  )
print(  "\nPUBLIC VALIDATION SCORE:  ", 
        roc_auc_score( y_test[:PUBLIC_CUTOFF], y_pred[:PUBLIC_CUTOFF] )  )
print(  "\nPRIVATE VALIDATION SCORE: ",
        roc_auc_score( y_test[PUBLIC_CUTOFF:], y_pred[PUBLIC_CUTOFF:] )  )

final_sub.to_csv("val_krishnas_r_lgb_bag3.csv", index=False, float_format='%.8f')