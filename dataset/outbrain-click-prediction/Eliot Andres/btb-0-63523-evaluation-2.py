import pandas as pd
import numpy as np 

reg = 10 # trying anokas idea of regularization
eval = True

train = pd.read_csv("../input/clicks_train.csv")
# We are going to find the amount of click by document id
clicked = train[train.clicked==1]
clicked.ad_id.values
promoted = pd.read_csv("../input/promoted_content.csv")
clicked_documents = promoted[promoted.ad_id.isin(clicked.ad_id.values)]
documentClicks = clicked_documents.document_id.value_counts()
clickNormalized = documentClicks / documentClicks.values.sum()
def getAlternativeScore(adId):
    documentId = promoted[promoted.ad_id == adId].document_id.values[0]
    if documentId in clickNormalized.index:
        return clickNormalized[documentId]
    return 0
    

if eval:
	ids = train.display_id.unique()
	ids = np.random.choice(ids, size=len(ids)//10, replace=False)

	valid = train[train.display_id.isin(ids)]
	train = train[~train.display_id.isin(ids)]
	
	print (valid.shape, train.shape)

cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()
del train

def get_prob(k):
    if k not in cnt:
        return getAlternativeScore(k)
    return cnt[k]/(float(cntall[k]) + reg)

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str,ad_ids))
   
if eval:
	from ml_metrics import mapk
	
	y = valid[valid.clicked==1].ad_id.values
	y = [[_] for _ in y]
	p = valid.groupby('display_id').ad_id.apply(list)
	p = [sorted(x, key=get_prob, reverse=True) for x in p]
	
	print (mapk(y, p, k=12))
else:
	subm = pd.read_csv("../input/sample_submission.csv") 
	subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
	subm.to_csv("subm_reg_1.csv", index=False)

