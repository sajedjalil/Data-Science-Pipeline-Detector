import pandas as pd
import numpy as np 

reg = 10 # trying anokas idea of regularization

train = pd.read_csv("../input/clicks_train.csv")

cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()
print ("1")
def get_prob(k):
    if k not in cnt:
        return 0
    return cnt[k]/(float(cntall[k]) + reg)

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str,ad_ids))
print ("1")
subm = pd.read_csv("../input/sample_submission.csv") 
print ("1")
subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
print ("1")
subm.to_csv("subm_reg.csv", index=False)