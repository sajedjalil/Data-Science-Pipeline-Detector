import pandas as pd
import numpy as np
import time as timer
import xgboost as xgb
import itertools as iter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing 


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)

def feature_corr(groups, x, test):
	for field in groups:
		corr_df = x[field].corr('pearson')
		corr_df = corr_df[corr_df.abs()>0.95]
		corr_df = corr_df.where((np.tril(np.ones(corr_df.shape))- np.eye(len(corr_df.columns))).astype(np.bool))
		corr_df = corr_df.stack().reset_index()
		for i in range(len(corr_df.index)):
			col1 = corr_df.get_value(i,'level_0')
			col2 = corr_df.get_value(i,'level_1')
			x['{}_{}_sub'.format(col1, col2)] = x[col1] - x[col2]
			x['{}_{}_add'.format(col1, col2)] = x[col1] + x[col2]
			test['{}_{}_sub'.format(col1, col2)] = test[col1] - test[col2]
			test['{}_{}_add'.format(col1, col2)] = test[col1] + test[col2]
	return (x, test)


def date_processing(X):
	X['Date'] = pd.to_datetime(X['Original_Quote_Date'])
	X.drop('Original_Quote_Date', axis=1, inplace = True)
	X['Date_Year'] = X['Date'].apply(lambda x: int(str(x)[:4]))
	X['Date_Month'] = X['Date'].apply(lambda x: int(str(x)[5:7]))
	X['Date_Weekday'] = X['Date'].dt.dayofweek     	#The day of the week with Monday=0, Sunday=6
	X.drop('Date', axis= 1, inplace = True)
	return X
	
	
def feature_importance_rf(x, y):
	model_rf = RandomForestClassifier()
	model_rf.fit(x,y)
	# print(x_values.columns.shape)
	N = len(model_rf.feature_importances_)
	#print(model_rf.feature_importances_)
	indxs = np.argsort(model_rf.feature_importances_)[:10]
	return indxs
	
def classify_xgboost( x, y, test):	
	model = xgb.XGBClassifier(
	learning_rate =0.25,
	n_estimators=101,
	max_depth=6,
	min_child_weight=3,
	subsample=0.86,
	colsample_bytree=0.77,
	objective= 'binary:logistic',
	scale_pos_weight=211879/48894,
	seed = 42,
	silent = False
	)
	# skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
	# for train_index, val_index in skf.split(x, y):
		# x_train, x_val = x.iloc[train_index], x.iloc[val_index]
		# y_train, y_val = y[train_index], y[val_index]
		# model.fit(x_train, y_train, early_stopping_rounds=50, eval_metric='auc',
        # eval_set=[(x_val, y_val)])
	model.fit(x, y, eval_metric='auc')
	y_pred = model.predict_proba(test)
	pred = y_pred[:,1]
	print(pred)
	return pred

def output(y_pred):
    out_sub = pd.read_csv('../input/sample_submission.csv')
    out_sub.QuoteConversion_Flag = y_pred
    filename = 'benchmark.csv'
    out_sub.to_csv(filename, index=False)


if __name__ == '__main__':
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
    
    le = preprocessing.LabelEncoder()
       
    dt.fillna(-1, inplace = True)
    dt_test.fillna(-1, inplace = True)
    dt['missing'] = dt[dt == -1].count(axis = 1)
    dt_test['missing'] = dt_test[dt_test == -1].count(axis = 1)
    x_cat = dt.select_dtypes(include = ['object'])
    for cols in x_cat.columns:
    	le.fit(np.unique(list(dt[cols].values)+ list(dt_test[cols].values)))
    	dt[cols] = le.transform(list(dt[cols].values))
    	dt_test[cols] = le.transform(list(dt_test[cols].values))
    	
    	
    for field in sorting_fields:
    		groups.append([col for col in dt.columns if col.startswith(field)])
    
    x ,test = feature_corr(groups, dt, dt_test)

    feature_indxs = feature_importance_rf(dt,y)
    top_feat = dt.columns[feature_indxs]
    print(top_feat)

    for pair in list(iter.combinations(top_feat, 2)):
    	col1 = list(pair)[0]
    	col2 = list(pair)[1]
    	x['{}_{}_add'.format(col1, col2)] = dt[col1] + dt[col2]
    	test['{}_{}_add'.format(col1, col2)] = dt_test[col1] + dt_test[col2]
    
    for cols in x.columns:
    	if x[cols].std() <= 0.05:
    		print(cols)
    		x.drop(cols, axis = 1, inplace = True)
    		test.drop(cols, axis = 1, inplace = True)
    
    x = x.apply(lambda col: (col - np.mean(col))/np.std(col))
    test = test.apply(lambda col: (col - np.mean(col))/np.std(col))
    print(x.shape)
    
    y_pred = classify_xgboost(x, y, test)
    output(y_pred)
    t1 = timer.time()