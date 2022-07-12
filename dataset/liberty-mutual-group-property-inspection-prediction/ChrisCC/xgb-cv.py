# Baseline: 0.373090  'model__min_child_weight': [1],
				#   'model__subsample': [0.8],
				#   'model__max_depth': [5],
				#   'model__learning_rate': [0.05],
				#   'model__n_estimators': [200]
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
	est = pipeline.Pipeline([
								#('fs', SelectKBest(score_func=f_classif,k=610)),
								#('sc', StandardScaler()),
								('model', xgb.XGBRegressor()
								 #XGBoostRegressor2(params=params, offset=5000, num_rounds=2000)
								)
							])


	# Create a parameter grid to search for best parameters for everything in the pipeline
	param_grid = {'model__n_estimators': [200],
				  'model__learning_rate': [0.01],
				  'model__subsample': [0.5, 1.0],
				  'model__colsample_bytree': [0.4, 1.0],
				  'model__min_child_weight': [1],
				  'model__max_depth': [5],
				  }

	# Normalized Gini Scorer
	gini_scorer = metrics.make_scorer(Gini, greater_is_better = True)

	# Initialize Grid Search Model
	model = grid_search.GridSearchCV(estimator  = est,
									 param_grid = param_grid,
									 scoring	= gini_scorer,
									 verbose	= 10,
									 n_jobs	 = -1,
									 iid		= True,
									 refit	  = True,
									 cv		 = 2)
	# Fit Grid Search Model
	model.fit(trainX, trainY)
	print("Best score: %0.3f" % model.best_score_)
	print("Best parameters set:")
	best_parameters = model.best_estimator_.get_params()
	for param_name in sorted(param_grid.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

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
	testDF = testDF.set_index('Id')
	testDF.to_csv('GBR.csv')
	return


if __name__ == '__main__':

	start = time.time()
	print ("Starting...")
	model = search_model()
	submit(model)
	print ("Finished in %0.3fs" % (time.time() - start))