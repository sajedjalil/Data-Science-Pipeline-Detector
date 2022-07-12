
import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.cross_validation import StratifiedShuffleSplit

def xgboost_pred(train,labels,test):
	params = {}
	params["objective"] = "reg:linear"
	params["eta"] = 0.01
	params["min_child_weight"] = 20
	params["subsample"] = 0.8
	params["colsample_bytree"] = 0.8
	params["scale_pos_weight"] = 1
	params["seed"] = 369
	params["silent"] = 1
	params["max_depth"] = 9
    
    
	plst = list(params.items())

	#Using 5000 rows for early stopping. 
	offset = 1000

	num_rounds = 10000
	xgtest = xgb.DMatrix(test)
	labelss = np.log(labels)
    
	#create a train and validation dmatrices 
	
	xgtrain = xgb.DMatrix(train[offset:,:], label=labelss[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labelss[:offset])

	#train using early stopping and predict
	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
	preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)


	#reverse train and labels and use different 5k for early stopping. 
	# this adds very little to the score but it is an option if you are concerned about using all the data. 
	train = train[::-1,:]
	labels = np.log(labels[::-1])

	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	watchlist = [(xgtrain, 'train'),(xgval, 'val')]
	model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
	preds2 = model.predict(xgtest,ntree_limit=model.best_iteration)


	#combine predictions
	#since the metric only cares about relative rank we don't need to average
	preds = (np.exp(preds1)*0.5) + (np.exp(preds2)*0.5)
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

print(columns)
train_s = np.array(train_s)
test_s = np.array(test_s)

# label encode the categorical variables
for i in range(train_s.shape[1]):
    if i in [3,4,5,6,7,8,10,11,14,15,16,19,21,27,28,29]:
        print(i)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
        train_s[:,i] = lbl.transform(train_s[:,i])
        test_s[:,i] = lbl.transform(test_s[:,i])

train_s = train_s.astype(float)
test_s = test_s.astype(float)


#preds = xgboost_pred(train_s,labels,test_s)

#generate solution
#preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
#preds = preds.set_index('Id')
#preds.to_csv('xgb_v1.csv')