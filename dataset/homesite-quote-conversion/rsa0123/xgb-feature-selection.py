import pandas as pd
import numpy as np
import time as timer
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing 

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)


def date_processing(X):
	X['Date'] = pd.to_datetime(X['Original_Quote_Date'])
	X.drop('Original_Quote_Date', axis=1, inplace = True)
	X['Date_Year'] = X['Date'].apply(lambda x: int(str(x)[:4]))
	X['Date_Month'] = X['Date'].apply(lambda x: int(str(x)[5:7]))
	X['Date_Weekday'] = X['Date'].dt.dayofweek     	#The day of the week with Monday=0, Sunday=6
	X.drop('Date', axis= 1, inplace = True)
	return X

	
def feature_processing(X):
	le = preprocessing.LabelEncoder()
	
        
	X.fillna(-1, inplace = True)
	x_cat = X.select_dtypes(include = ['object'])
	for cols in x_cat.columns:
	    le.fit(list(X[cols].values))
	    X[cols] = le.transform(list(X[cols].values))
	#X[x_cat.columns] = X[x_cat.columns].apply(le.fit_transform)
	for  cols in X.columns:
		if X[cols].std() == 0:
			X.drop(cols, axis = 1, inplace = True)
	x_numeric = X.select_dtypes(exclude = ['object'])
	X[x_numeric.columns] = X[x_numeric.columns].apply(lambda col: (col - np.mean(col))/np.std(col))
	X[x_cat.columns] = X[x_cat.columns].apply(lambda col: (col - np.mean(col))/np.std(col))
	return X

	
def feature_importance_rf(x, y):
	model_rf = RandomForestClassifier()
	model_rf.fit(x,y)
	# print(x_values.columns.shape)
	N = len(model_rf.feature_importances_)
	indxs = np.argsort(model_rf.feature_importances_)[:250]
	return indxs
	
	

t0 = timer.time()

y_field = 'QuoteConversion_Flag'

irrelevant_fields =['QuoteNumber', y_field]
sorting_fields = ['Field','Coverage','Sales','Personal','Property','Geographic','Date']
groups = []

y = train[y_field].values

train.drop(irrelevant_fields,axis = 1, inplace = True)
test.drop(['QuoteNumber'],axis = 1, inplace = True)

dt = date_processing(train)
dt_test = date_processing(test)
x = feature_processing(dt)
x_test = feature_processing(dt_test)

feature_indxs = feature_importance_rf(x,y)
	
x_train = x[feature_indxs]
y_train = y
x_test = x_test[feature_indxs]	

print(x_train)	

params = {
'learning_rate' : 0.25, 
'n_estimators' : 25, 
'max_depth' : 6,
'min_child_weight' : 3,
'subsample' : 0.83,
'colsample_bytree' : 0.77,
'objective' : 'binary:logistic',
'scale_pos_weight' : 1,
'seed' : 42 }

model = xgb.XGBClassifier(**params)
model.fit(x_train, y_train)
y_pred = model.predict_proba(x_test)[:,1]
out_sub = pd.read_csv('../input/sample_submission.csv')
out_sub.QuoteConversion_Flag = y_pred
filename = 'benchmark.csv'
out_sub.to_csv(filename, index=False)

t1 = timer.time()