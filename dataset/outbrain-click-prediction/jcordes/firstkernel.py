# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

reg = 10 # trying anokas idea of regularization
eval = True # True = split off 10% of training data for validation and test performance

train = pd.read_csv("../input/clicks_train.csv")

if eval:
	ids = train.display_id.unique()
	ids = np.random.choice(ids, size=len(ids)//10, replace=False)

	valid = train[train.display_id.isin(ids)] # random 10% for validation data 
	train = train[~train.display_id.isin(ids)] # remaining 90% as training data
	
	print (valid.shape, train.shape)

cnt = train[train.clicked==1].ad_id.value_counts() # group # of clicks by ad 
cntall = train.ad_id.value_counts() # group # of displays by ad
del train

def get_prob(k):
    if k not in cnt:
        return 0
    return cnt[k]/(float(cntall[k]) + reg)  # return the proportion of ad clicks / displays

def srt(x):
    ad_ids = map(int, x.split()) # take in list of ads shown to each user
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True) # re-sort the ads by training clicks / displays
    return " ".join(map(str,ad_ids)) # return the list with the ads sorted for submission
   
if eval:
	from ml_metrics import mapk
	
	y = valid[valid.clicked==1].ad_id.values # create list of ad click counts in validation set
	y = [[_] for _ in y]
	p = valid.groupby('display_id').ad_id.apply(list)
	p = [sorted(x, key=get_prob, reverse=True) for x in p] # create list in order expected
	
	print (mapk(y, p, k=12)) # compare predicted order vs. actual order in validation set
else:
	subm = pd.read_csv("../input/sample_submission.csv") # load the sample submission file
	subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x)) # re-sort the ads by overall training clicks / display
	subm.to_csv("subm_reg_1.csv", index=False) 
