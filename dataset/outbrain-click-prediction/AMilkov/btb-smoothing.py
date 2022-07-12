import pandas as pd
import numpy as np 

train = pd.read_csv("../input/clicks_train.csv")
cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()
del train

def get_prob(k):
    return cnt[k]/(float(cntall[k]) + 5) if k in cnt else -1

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str, ad_ids))
   
subm = pd.read_csv("../input/sample_submission.csv") 
subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
subm.to_csv("subm_reg_1.csv", index=False)

