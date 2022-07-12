import pandas as pd
import numpy as np 

reg = 10

print('Reading training data...')
train = pd.read_csv("../input/clicks_train.csv")

cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()

def get_prob(k):
    if k not in cnt:
        return 0
    return cnt[k] ** 2 / (float(cntall[k]) + reg)

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str,ad_ids))

print('Reading test data...') 
subm = pd.read_csv("../input/sample_submission.csv")

print('Preparing submission file..')
subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
subm.to_csv("subm_1prob_reg10.csv", index=False)
print('Submission file ready...')