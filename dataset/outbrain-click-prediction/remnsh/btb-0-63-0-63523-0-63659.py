import pandas as pd
import numpy as np

train = pd.read_csv("../input/clicks_train.csv")

cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()
ave_ctr = np.sum(cnt)/(float(np.sum(cntall)))

def get_prob(k):
    if k in cnt:
        return cnt[k] / (float(cntall[k]) + 10)
    else:
        if k in cntall:
            # use -imps for 0 click penalty
            return -1 * cntall[k]
        else:
            # use average value for no imp ad
            return ave_ctr

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str,ad_ids))

subm = pd.read_csv("../input/sample_submission.csv")
subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
subm.to_csv("subm_reg_2.csv", index=False)