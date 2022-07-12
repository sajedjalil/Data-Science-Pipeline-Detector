# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#59714 is the output length of this full thing!
Ntrain,Ntest,Nsub =87141731,32225162,6245533


# Any results you write to the current directory are saved as output.
print ('Thanks to the1owl,franckjay for lots of ideas:')
print ('https://www.kaggle.com/the1owl/outbrain-click-prediction/to-click-or-not-to-click-that-is-the-question/discussion')
print ('https://www.kaggle.com/franckjay/outbrain-click-prediction/easy-random-forests')

testing=False
filename = 'TREE_n10_split4_leaf2_TEST'
print ("Get tables to combine")
categories = pd.read_csv('../input/documents_categories.csv')
#entities = pd.read_csv('../input/documents_entities.csv')
#meta = pd.read_csv('../input/documents_meta.csv')
topics = pd.read_csv('../input/documents_topics.csv')
content = pd.read_csv('../input/promoted_content.csv')
#events = pd.read_csv('../input/events.csv')[['display_id','document_id','platform']]

#views = pd.read_csv('../input/page_views_sample.csv')
#views['geo_location'] = pd.factorize(views['geo_location'].str[:2])[0]

print ("categories",categories.columns,categories.shape)
#print ("entities",topics.columns,entities.shape)
#print ("meta",topics.columns,meta.shape)
print ("topics",topics.columns,topics.shape)
print ("content",content.columns,content.shape)
#print ("views",views.columns,content.shape)

#print ("events",events.columns,events.shape)
def merging(chunk):
    global content,events,topics,categories
    
    chunk=chunk.merge(content,how='inner',on=['ad_id'])
    #print ("chunk content",chunk.columns,chunk.shape)
    
    # content=content.merge(events,
    #     how='inner',on='document_id')
    # print ("chunk events",chunk.columns,chunk.shape)
    
    #chunk=chunk.merge(topics,how='inner',on='document_id')
    #print ("chunk topics",chunk.columns,content.shape)

    #chunk=chunk.merge(categories,how='inner',on='document_id')
    
    # chunk=chunk.merge(views[['document_id', 'timestamp', 'platform', 'geo_location',
    #   'traffic_source']],how='inner',on='document_id')
    #print ("chunk categories",chunk.columns,chunk.shape)
    return chunk

print('Done combining')

chunksize=500000# Out of 87141731.
rand_state = 12345
params = {
            #'nthread': 3,
            'seed': rand_state,
            #'colsample_bytree': 0.8,
            'silent': 1,
            #'subsample': 0.85,
            'learning_rate': 0.1,
            'objective': 'reg:linear',
            'max_depth': 12,
            'gamma': 2,
            'min_child_weight': 4,
            'booster': 'gbtree'
            }
model = None

train = pd.read_csv('../input/clicks_train.csv',iterator=True,chunksize=chunksize) #Load data
print( 'Training')
total = 1
for chunk in train:
	print (total,total*chunksize)
	chunk=merging(chunk)
	#print (chunk.shape,chunk.columns)
	#exit()
	
	predictors=[x for x in chunk.columns if x not in ['display_id','ad_id','clicked']]
	chunk=chunk.fillna(0.0)
	train_x=chunk[predictors].values
	train_y=chunk["clicked"].values
	#alg = RandomForestClassifier(random_state=1, n_estimators=5, min_samples_split=4, min_samples_leaf=2, warm_start=True)
	#alg.fit(chunk[predictors], chunk["clicked"])#Fit the Algorithm
	dtrain = xgb.DMatrix(train_x, label=train_y)
	if total == 1:
	    print (chunk.columns)
	    print (train_x.shape, train_y.shape)
	    model=xgb.train(params, dtrain, 20)
	else:
	    scores_val = model.predict(xgb.DMatrix(train_x))
	    cv_score = mean_absolute_error(train_y, scores_val)
	    print (cv_score)
	    model.update(dtrain,10)
	    
	#print (model)
	    
	total =total+1
	if total >20:
		break
	train_x = None
	train_y = None
	chunk = None
	gc.collect()
    
train=''
print('Testing')
test= pd.read_csv('../input/clicks_test.csv',iterator=True,chunksize=chunksize) #Load data
predY=[]
total = 1
for chunk in test:
    print (total,total*chunksize)
    total =total+1
    chunk=merging(chunk)
    chunk=chunk.fillna(0.0)
    chunk_pred=list(model.predict(xgb.DMatrix(chunk[predictors].values)))
    predY += chunk_pred
    #if testing:
    #break
print('Done Testing')

print('Preparing for Submission')	
test=''#We do not want the iterable version of test
test= pd.read_csv('../input/clicks_test.csv')#But rather the full version
results=pd.concat(	(test,pd.DataFrame(predY)) ,axis=1,ignore_index=True,)#Combine the predicted values with the ids
print(results.head(10))
results.columns = ['display_id','ad_id','clicked']#Rename the columns
#results=results[results['clicked'] > 0.0]
results = results.sort_values(by=['display_id','clicked'], ascending=[True, False])
results = results.reset_index(drop=True)
results=results[['display_id','ad_id']].groupby('display_id')['ad_id'].agg(lambda col: ' '.join(map(str,col)))
#results.columns=[['display_id','ad_id']]
results.to_csv(filename+'.csv')	#
	
	