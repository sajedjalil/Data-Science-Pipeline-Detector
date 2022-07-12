#Initial KERNEL https://www.kaggle.com/tunguz!
import pandas as pd

import pandas as pd
import numpy  as np
from scipy.special import expit, logit
almost_zero = 1e-5 # 0.00001
almost_one  = 1-almost_zero # 0.99999


import os
print(os.listdir("../input"))

scores = {}

# Gated Recurrent Unit + Convolutional Neural Network
df = pd.read_csv("../input/weighted-app-chanel-os/subnew.csv",index_col="click_id").rename(columns={'is_attributed': 'FIRST'}) # 0.80121
scores["FIRST"] = 0.9565 #0.80121

#df["weighted"] = pd.read_csv("../input/weighted-app-chanel-os/subnew.csv")['is_attributed'].values # 0.7947
#scores["weighted"] = 0.9565

#df["single_xgb"] = pd.read_csv("../input/single-xgboost-in-r-histogram-optimized/sub_xgb_hist_R_50m.csv")['is_attributed'].values # 0.7959
#scores["single_xgb"] = 0.9686 

df["pranav"] = pd.read_csv("../input/single-model-by-pranav-hitting-0-9701/lightgbm_r.csv")['is_attributed'].values # 0.7959
scores["pranav"] = 0.9686 

#df["lgbm_count"] = pd.read_csv(  '../input/lightgbm-with-count-features/sub_lgb_balanced99.csv')['is_attributed'].values # 0.81177
#scores["lgbm_count"] = 0.9667

#df["deep"] = pd.read_csv('../input/deep-learning-support-9663/dl_support.csv')['is_attributed'].values # 0.81177
#scores["deep"] = 0.9663

df["smaller"] = pd.read_csv('../input/lightgbm-smaller/submission.csv')['is_attributed'].values # 0.81177
scores["smaller"] = 0.9678

df["donot"] = pd.read_csv('../input/do-not-congratulate/sub_mix_logits_ranks.csv')['is_attributed'].values # 0.81177
scores["donot"] = 0.9695

#df["talking"] = pd.read_csv('../input/rank-averaging-on-talkingdata/rank_averaged_submission.csv')['is_attributed'].values # 0.81177
#scores["talking"] = 0.9689

df["anttips"] = pd.read_csv('../input/anttip-s-wordbatch-fm-ftrl-9752-version/wordbatch_fm_ftrl.csv')['is_attributed'].values # 0.81177
scores["anttips"] = 0.9750

df["max"] = pd.read_csv('../input/sub-stacked/sub_stacked_max.csv')['is_attributed'].values # 0.81177
scores["max"] = 0.9778





weights = [0] * len(df.columns)
power = 128
dic = {}

for i,col in enumerate(df.columns):
    weights[i] = scores[col] ** power
    dic[i] = df[col].clip(almost_zero,almost_one).apply(logit) * weights[i] 
    
print(weights[:])
totalweight = sum(weights)

temp = []
for x in dic:
    if x == 0:
        temp = dic[x]
    else:
        temp = temp+dic[x]

temp = temp/(totalweight)

df['is_attributed'] = temp
df['is_attributed'] = df['is_attributed'].apply(expit)
df['is_attributed'].to_csv("ensembling.csv", index=True, header=True)