import pandas as pd
import numpy as np 


train = pd.read_csv("../input/clicks_train.csv")

cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()
del train

def get_prob(k):
    if k not in cnt and k in cntall:
        return 0.05
    if k not in cnt:
        return 1
    return cnt[k]/(float(cntall[k]))

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str,ad_ids))

subm = pd.read_csv("../input/sample_submission.csv") 
subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
subm.to_csv("subm_reg_1.csv", index=False)

