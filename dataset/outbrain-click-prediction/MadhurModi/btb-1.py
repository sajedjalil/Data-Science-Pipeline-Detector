import pandas as pd
import numpy as np 
eps = 5
print('step 0.1')
train = pd.read_csv("../input/clicks_train.csv")
print('step 0.2')
cnt = train[train.clicked==1].ad_id.value_counts()
print('step 0.3')
cntall = train.ad_id.value_counts()
print('step 0.4')
def get_prob(k):
    if k not in cnt:
        return 0
    return cnt[k] ** 2 / (float(cntall[k]) + eps)

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str,ad_ids))
   
subm = pd.read_csv("../input/sample_submission.csv") 
print('step 1')
subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
print('step 2')
subm.to_csv("subm_1prob_eps110.csv", index=False)
print('step last')