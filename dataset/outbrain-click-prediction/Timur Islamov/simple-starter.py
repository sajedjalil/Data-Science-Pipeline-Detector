import pandas as pd

train = pd.read_csv("../input/clicks_train.csv")

clicked_counts = dict(train[train.clicked == 1].ad_id.value_counts().astype(float))
all_counts = dict(train.ad_id.value_counts().astype(float))


reg = 4


def get_prob(k):
    return clicked_counts[k] / (all_counts[k] + reg) if k in clicked_counts else 0

def sort_ad_ids(v):
    ad_ids = map(int, v.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str, ad_ids))


submission = pd.read_csv("../input/sample_submission.csv")
submission.ad_id = submission.ad_id.map(sort_ad_ids)
submission.to_csv("simple_reg_{}.csv.gz".format(reg), index=False, compression="gzip")