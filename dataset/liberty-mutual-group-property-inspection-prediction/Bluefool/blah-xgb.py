'''


Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@Soutik

Have fun;)
'''

import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer


def xgboost_pred(train,labels,test):
	params = {}
	params["objective"] = "reg:linear"
	params["eta"] = 0.01
	params["min_child_weight"] = 1
	params["scale_pos_weight"] = 1
	params["silent"] = 1
	params["max_depth"] = 6
	params["seed"] = 369
    
    
	plst = list(params.items())

	#Using 5000 rows for early stopping. 
	offset = 4000

	num_rounds = 10000
	xgtest = xgb.DMatrix(test)
	labels = np.log1p(labels[::-1])

	#create a train and validation dmatrices 
	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	#train using early stopping and predict
	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
	preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)

	preds = np.expm1(preds1) 
	return preds

#load train and test 
train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)


labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

train_s = train
test_s = test


columns = train.columns
test_ind = test.index


train_s = np.array(train_s)
test_s = np.array(test_s)

# label encode the categorical variables
for i in range(train_s.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
    train_s[:,i] = lbl.transform(train_s[:,i])
    test_s[:,i] = lbl.transform(test_s[:,i])

train_s = train_s.astype(float)
test_s = test_s.astype(float)


preds1 = xgboost_pred(train_s,labels,test_s)

#model_2 building

train = train.T.to_dict().values()
test = test.T.to_dict().values()

vec = DictVectorizer()
train = vec.fit_transform(train)
test = vec.transform(test)

preds2 = xgboost_pred(train,labels,test)


preds = 0.5 * preds1 + 0.5 * preds2

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_v1.csv')