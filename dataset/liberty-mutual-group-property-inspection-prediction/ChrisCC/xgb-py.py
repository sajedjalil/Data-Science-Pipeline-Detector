import time
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble
from sklearn import feature_extraction
from sklearn import pipeline, metrics, grid_search
from sklearn.utils import shuffle
import xgboost as xgb

def load_data():
	DV = feature_extraction.DictVectorizer(sparse=False)
	trainData=pd.read_csv("../input/train.csv")
	testData=pd.read_csv("../input/test.csv")
	catCols=['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12', 'T2_V13']


	trainY=trainData['Hazard'].values
	trainNumX=trainData.drop(catCols + ['Hazard','Id'], axis=1)
	trainCatX=trainData[catCols]
	trainCatVecX = DV.fit_transform(trainCatX.T.to_dict().values())
	trainX = np.hstack((trainCatVecX,trainNumX))


	testNumX=testData.drop(catCols + ['Id'], axis=1)
	testCatX=testData[catCols]
	testCatVecX = DV.fit_transform(testCatX.T.to_dict().values())
	testX = np.hstack((testCatVecX,testNumX))
	testId = testData['Id'].values

	return trainX, trainY, testX, testId

def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)	
	
class XGBoostRegressor():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'reg:linear'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self	

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true


def search_model():
	trainX, trainY, _, _ = load_data()
	est = XGBoostRegressor(silent = 1)

	# Create a parameter grid to search for best parameters for everything in the pipeline
	parameters = {
	'num_boost_round': [100],
	'eta': [0.05, 0.1, 0.3],
	'max_depth': [6, 9, 12],
	'subsample': [0.9, 1.0],
	'colsample_bytree': [0.9, 1.0],
	}

	# Normalized Gini Scorer
	gini_scorer = metrics.make_scorer(Gini, greater_is_better = True)
	
	print ("Start searching model...")
	# Initialize Grid Search Model
	model = grid_search.GridSearchCV(est, parameters,scoring=gini_scorer, n_jobs=1, cv=2)

	# Fit Grid Search Model
	model.fit(trainX, trainY)
	print("Best score: %0.3f" % model.best_score_)
	print("Best parameters set:")
	best_parameters = model.best_estimator_.get_params()
	#for param_name in sorted(parameters.keys()):
	#	print("\t%s: %r" % (parameters, best_parameters[param_name]))

	# Get best model
	best_model = model.best_estimator_

	# Fit model with best parameters optimized for normalized_gini
	best_model.fit(trainX,trainY)
	return best_model	

def submit(model):
    # load test data
    _, _, testX, testId = load_data()
    testY = model.predict(testX)
    testDF = pd.DataFrame({"Id": testId, "Hazard": testY})
    testDF.to_csv('GBR.csv')
    return


if __name__ == '__main__':

	start = time.time()
	print ("Starting...")
	model = search_model()
	submit(model)
	print ("Finished in %0.3fs" % (time.time() - start))
	
