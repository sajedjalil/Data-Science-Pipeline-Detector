import pandas as pd
import numpy as np 

reg = 10 # trying anokas idea of regularization
eval = True

dtypes = {'ad_id': np.float32, 'clicked': np.int8}

print('starting ...')
train = pd.read_csv('../input/clicks_train.csv', nrows=10000, usecols=['ad_id','clicked'])

ad_likelihood = train.groupby('ad_id').clicked.agg(['count','sum','mean']).reset_index()
M = train.clicked.mean()
del train

ad_likelihood['likelihood'] = ad_likelihood['count'] 

test = pd.read_csv("../input/clicks_test.csv", nrows=10000)
test = test.merge(ad_likelihood, how='left')
test.likelihood.fillna(M, inplace=True)

test = pd.read_csv("../input/clicks_test.csv")
test = test.merge(ad_likelihood, how='left')
test.likelihood.fillna(M, inplace=True)

test.sort_values(['display_id','likelihood'], inplace=True, ascending=False)
subm = test.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()
print(subm)
#subm.to_csv("subm.csv", index=False)

'''

cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()
del train

def get_prob(k):
    if k not in cnt:
        return 0
    return cnt[k]/(float(cntall[k]) + reg)

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str,ad_ids))
   
if eval:
	from ml_metrics import mapk
	
	y = valid[valid.clicked==1].ad_id.values
	y = [[_] for _ in y]
	print(y)
	p = valid.groupby('display_id').ad_id.apply(list)
	p = [sorted(x, key=get_prob, reverse=True) for x in p]
	print(p)
	
	print (mapk(y, p, k=12))
else:
	subm = pd.read_csv("../input/sample_submission.csv") 
	subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
	subm.to_csv("subm_reg_1.csv", index=False)



train = pd.read_csv("../input/clicks_train.csv", usecols=['ad_id','clicked'], dtype=dtypes)

ad_likelihood = train.groupby('ad_id').clicked.agg(['count','sum','mean']).reset_index()
M = train.clicked.mean()
del train

ad_likelihood['likelihood'] = (ad_likelihood['sum'] + 12*M) / (12 + ad_likelihood['count'])

test = pd.read_csv("../input/clicks_test.csv")
test = test.merge(ad_likelihood, how='left')
test.likelihood.fillna(M, inplace=True)

test.sort_values(['display_id','likelihood'], inplace=True, ascending=False)
subm = test.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()
subm.to_csv("subm.csv", index=False)
'''

