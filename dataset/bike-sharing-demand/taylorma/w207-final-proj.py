import pandas as pd
#import sklearn as sk
from sklearn.linear_model import LinearRegression


def get_feature_mat(fname):
	#feature engineering in this funciton is applied to both test and train
	df 	= pd.read_csv("../input/"+fname)
	return(df)

train, test = [get_feature_mat(fname) for fname in ['train.csv', 'test.csv']]
print('\nSummary of train dataset:\n')
print(train.describe())
print('\nSummary of test dataset:\n')
print(test.describe())

#print(train.columns.values)
#print(test.columns.values)

train_data = train[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed']]
train_cnt, train_casual, train_registered = train[['count']], train[['casual']], train[['registered']]

test_data = test[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed']]
# Fit linear regression model with training data
lr_count = LinearRegression(fit_intercept=True)
lr_count.fit(train_data, train_cnt)

lr_casual = LinearRegression(fit_intercept=True)
lr_casual.fit(train_data, train_cnt)

lr_registered = LinearRegression(fit_intercept=True)
lr_registered.fit(train_data, train_cnt)
