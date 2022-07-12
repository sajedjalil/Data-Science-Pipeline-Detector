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
import xgboost as xgb

train = pd.read_csv("../input/train.csv",sep=",",index_col=0)
submit_X = pd.read_csv("../input/test.csv",sep=",",index_col=0)

data_Y = train['TARGET'].copy()
data_X = train.copy()
del data_X['TARGET']
	
# Drop constant features
features_to_drop = []
for f in data_X.columns:
	if len(data_X[f].value_counts())==1:
		features_to_drop.append(f)
print(len(features_to_drop))
data_X = data_X.drop(features_to_drop,axis=1)
submit_X = submit_X.drop(features_to_drop,axis=1)

# drop equal features	
import itertools
combn2 = itertools.combinations(data_X.columns, 2)
features_to_drop = []
for f1,f2 in combn2:
	if (f1 not in features_to_drop) & (f2 not in features_to_drop):
		if np.sum(data_X[f1].values-data_X[f2].values)==0:
			#print(f1+' '+f2)
			features_to_drop.append(f2)
print(len(features_to_drop))
data_X = data_X.drop(features_to_drop,axis=1)
submit_X = submit_X.drop(features_to_drop,axis=1)

features_to_drop = []
for f in data_X.columns:
	train_cnt = data_X[f].value_counts()/len(data_X)
	if (train_cnt.values[0]>0.99):
		features_to_drop.append(f)
print(len(features_to_drop))		
data_X = data_X.drop(features_to_drop,axis=1)
submit_X = submit_X.drop(features_to_drop,axis=1)

data_X["pct_comp"]=(data_X==0).astype(int).sum(axis=1)
submit_X["pct_comp"]=(submit_X==0).astype(int).sum(axis=1)

from sklearn.cross_validation import StratifiedKFold
skf = StratifiedKFold(data_Y.values,n_folds=5,shuffle=True,random_state=5)

from sklearn import metrics

param = {}
param['objective'] = 'binary:logistic'
param['silent'] = 1
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['booster'] = 'gbtree'
param['max_depth'] = 5
param['min_child_weight'] = 7
param['scale_pos_weight'] = 1
param['subsample'] = .9 
param['colsample_bytree'] = .6 
param['eta'] = 0.025
param['seed'] = 1
rounds = 301

i=1
results=[]
for train,test in skf:
	xg_train = xgb.DMatrix(data_X.values[train],data_Y.values[train])
	xg_valid = xgb.DMatrix(data_X.values[test],data_Y.values[test])
	watchlist = [ (xg_train,'train'),(xg_valid,'validation')]
	
	bst = xgb.train(param, xg_train, num_boost_round=rounds,evals=watchlist,verbose_eval=False)
	
	pred_Y = bst.predict(xg_valid)
	fpr, tpr, thresholds = metrics.roc_curve(data_Y.values[test], pred_Y)
	results.append(metrics.auc(fpr, tpr))
	#rounds.append(bst.best_ntree_limit)
	print("CV"+str(i)+" done")
	i+=1

print("RFC results : %.6f/%.6f ; min=%.6f/max=%.6f" 
    %(np.mean(results),np.std(results),np.min(results),np.max(results)))
		
xg_sub = xgb.DMatrix(submit_X)
xg_train = xgb.DMatrix(data_X,data_Y)
watchlist = [ (xg_train,'train') ]
bst = xgb.train(param, xg_train, num_boost_round=rounds,evals=watchlist,verbose_eval=False)

sub_pred = bst.predict(xg_sub)
submission = pd.DataFrame({'ID':submit_X.index.values,'TARGET':sub_pred})
submission.to_csv('xgb_submission.csv',index=False)