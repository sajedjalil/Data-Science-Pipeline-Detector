import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_friedman1
from sklearn import preprocessing 
from sklearn.ensemble import GradientBoostingRegressor
import pandas 
from sklearn import grid_search
from sklearn import cross_validation as cv
def main(): 
	data_train = pandas.read_csv("train.csv")
	data_test = pandas.read_csv("test.csv")
	predictors = data_train.columns.values 
	print(predictors)
	categorical_predictors = predictors[1:-15]
	categorical_predictors_to_remove = []
	temp_categorical_predictors = [] 
	for j in range(0,len(categorical_predictors)): 
		tester = True 
		i = categorical_predictors[j]
		for x in data_test[i].unique(): 
			if x not in data_train[i].unique(): 
				tester = False
		if tester == False: 
			print(i)
			categorical_predictors_to_remove.append(i)
	for i in categorical_predictors: 
		if i not in categorical_predictors_to_remove: 
			temp_categorical_predictors.append(i)
	categorical_predictors = temp_categorical_predictors
	predictors = predictors[-15:-1]
	predictors = np.append(predictors,categorical_predictors)
	print(predictors)
	for i in categorical_predictors: 
		print(i)
		le = preprocessing.LabelEncoder()
		le.fit(data_train[i])
		data_train[i] = le.transform(data_train[i])
		data_test[i] = le.transform(data_test[i])
	scaler = preprocessing.StandardScaler() 
	scaler.fit(data_train[predictors])
	data_train[predictors] = scaler.transform(data_train[predictors])
	data_test[predictors] = scaler.transform(data_test[predictors])
	
	parameters={'min_impurity_split':[1e-1],'learning_rate':[1e-1],'min_samples_split':[7],'verbose':[2],'max_depth':[7],'min_samples_leaf':[1],'subsample':[1.0],'loss':['ls'],'n_estimators':[100]}
	clf = grid_search.GridSearchCV(GradientBoostingRegressor(),parameters) 
	clf.fit(data_train[predictors],data_train['loss'])
	data_predictions = clf.predict(data_test[predictors])

	data_test['loss'] = data_predictions
	outputFrame = pandas.DataFrame()
	outputFrame['id'] = data_test['id']
	outputFrame['loss'] = data_test['loss']
	outputFrame.to_csv("output.csv",index =False)
main()