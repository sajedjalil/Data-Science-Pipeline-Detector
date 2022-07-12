import pandas as pd
import re
import numpy as np
from sklearn import preprocessing

train_file = '../input/train.csv'
test_file = '../input/test.csv'
output_file = 'predictions.csv'

def training():
	
	train = pd.read_csv(train_file)
	train = feature_engineering(train)
	test = pd.read_csv(test_file)
	test = feature_engineering(test)
	X_train, X_test = construct_matrix(train, test)
	label = train['Sales']
	
	from sklearn.ensemble import RandomForestRegressor as rfc
	clf = rfc(n_estimators=15)
	clf.fit(X_train, label)
	predicted = clf.predict(X_test)
	
	test['Sales'] = predicted
	test[[ 'Id', 'Sales' ]].to_csv(output_file)
	
def construct_matrix(train, test):
	category_feature_name = ['DayOfWeek','Year','Month','StoreType', 'Assortment', 'Promo']
	enc = preprocessing.LabelEncoder()
	X = []
	print('categorical to ind .................................')
	for fname in category_feature_name:
		t1 = train[fname].tolist()
		t2 = test[fname].tolist()
		print(t1[:10], t2[:10])
		binarized_t = enc.fit_transform(t1+t2)
		X.append(binarized_t)
	enc = preprocessing.OneHotEncoder()
	print('ind to binary ..........................')
	X = enc.fit_transform(np.array(X).transpose())

	X_train = X[:len(train)]
	X_test = X[len(train):]
	return X_train, X_test

def feature_engineering(data):
	data['Year'] =data['Date'].map(lambda t: re.split('\-',t)[0])
	data['Month'] =data['Date'].map(lambda t: re.split('\-',t)[1])
	
	store = pd.read_csv('../input/store.csv')
	data = pd.merge(data, store, on =['Store'], how = 'left')
	return data
	
if __name__ == '__main__':
	training()
	