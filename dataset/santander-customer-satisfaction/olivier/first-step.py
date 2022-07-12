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

# drop features without enough entropy
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
skf = StratifiedKFold(data_Y.values,n_folds=5,random_state=5)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=400, criterion='entropy', n_jobs=-1, bootstrap=True,
							min_samples_leaf=1, min_samples_split=2,
							max_features=40, max_depth=10, random_state=57)

from sklearn.cross_validation import cross_val_score
results = cross_val_score(rfc, data_X, data_Y, scoring='roc_auc', cv=skf, verbose=3, n_jobs=1)

print("RFC results : %.6f/%.6f ; min=%.6f/max=%.6f" 
    %(np.mean(results),np.std(results),np.min(results),np.max(results)))
rfc.fit(data_X,data_Y)
sub_pred = rfc.predict_proba(submit_X)[:,1]
submission = pd.DataFrame({'ID':submit_X.index.values,'TARGET':sub_pred})
submission.to_csv('rfc_submission.csv',index=False)
