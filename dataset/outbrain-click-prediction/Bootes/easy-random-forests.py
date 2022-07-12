# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#59714 is the output length of this full thing!
Ntrain,Ntest,Nsub =87141731,32225162,6245533


# Any results you write to the current directory are saved as output.
print ('Thanks to the1owl for lots of ideas:')
print ('https://www.kaggle.com/the1owl/outbrain-click-prediction/to-click-or-not-to-click-that-is-the-question/discussion')

testing=False
filename = 'TREE_n10_split4_leaf2_TEST'
print ("Get tables to combine")
content = pd.read_csv('../input/promoted_content.csv')
print('Done combining')

chunksize=50000# Out of 87141731.
train = pd.read_csv('../input/clicks_train.csv',iterator=True,chunksize=chunksize) #Load data
print( 'Training')
for chunk in train:
	chunk=pd.merge(chunk,content,how='left',on='ad_id')	
	predictors=[x for x in chunk.columns if x not in ['display_id','clicked']]
	chunk=chunk.fillna(0.0)
	alg = RandomForestClassifier(random_state=1, n_estimators=3, min_samples_split=4, min_samples_leaf=2, warm_start=True)
	alg.fit(chunk[predictors], chunk["clicked"])#Fit the Algorithm
	if testing:
		break

train=''
print('Testing')
test= pd.read_csv('../input/clicks_test.csv',iterator=True,chunksize=chunksize) #Load data
predY=[]
for chunk in test:
	init_chunk_size=len(chunk)
	chunk=pd.merge(chunk,content,how='left',on='ad_id')
	chunk=chunk.fillna(0.0)
	chunk_pred=list(alg.predict_proba(chunk[predictors]).astype(float)[:,1])
	predY += chunk_pred
	if testing:
		break
print('Done Testing')

print('Preparing for Submission')	
test=''#We do not want the iterable version of test
test= pd.read_csv('../input/clicks_test.csv')#But rather the full version
results=pd.concat(	(test,pd.DataFrame(predY)) ,axis=1,ignore_index=True)#Combine the predicted values with the ids
print(results.head(10))
results.columns = ['display_id','ad_id','clicked']#Rename the columns
#results=results[results['clicked'] > 0.0]
results = results.sort_values(by=['display_id','clicked'], ascending=[True, False])
results = results.reset_index(drop=True)
results=results[['display_id','ad_id']].groupby('display_id')['ad_id'].agg(lambda col: ' '.join(map(str,col)))
results.columns=[['display_id','ad_id']]
results.to_csv(filename+'.csv')	#
	
	