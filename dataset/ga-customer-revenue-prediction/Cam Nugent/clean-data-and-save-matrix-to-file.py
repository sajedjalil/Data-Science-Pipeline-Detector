#to speed up iteration on predictions, this script moves from the original data
#to a set of np arrays that are ready for import into machine learning algorithms
# you can tinker with how missing data arm imputed, which categorical columns are one-hot encoded
# etc. to try to arrive at a dataset that makes better predictions.

#X_train = np.load('X_train.dat')
#y_train = np.load('y_train.dat')
#X_test = np.load('X_test.dat')


import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from sklearn.preprocessing import LabelBinarizer

import gc
import json
import time

####
# import the train and test dataframes, flatten the json columns
####


test = pd.read_csv('../input/test.csv')

test.head()

#which columns have json
json_cols = ['device', 'geoNetwork', 'totals',  'trafficSource']

for column in json_cols:

	c_load = test[column].apply(json.loads)
	c_list = list(c_load)
	c_dat = json.dumps(c_list)

	test = test.join(pd.read_json(c_dat))
	test = test.drop(column , axis=1)

test.head()

train = pd.read_csv('../input/train.csv')


#which columns have json

for column in json_cols:

	c_load = train[column].apply(json.loads)
	c_list = list(c_load)
	c_dat = json.dumps(c_list)

	train = train.join(pd.read_json(c_dat))
	train = train.drop(column , axis=1)

train.head()

all_train = train
#all_train.head() 

final_test = test
#final_test.head()

submission = pd.read_csv('../input/sample_submission.csv')
#submission.head()

submission['fullVisitorId'] = submission['fullVisitorId'].astype('str')
final_test['fullVisitorId'] = final_test['fullVisitorId'].astype('str')

"""
####
# check submission length
####

#it is lower than the number of ids in the test set?
len(submission['fullVisitorId']) == len(set(submission['fullVisitorId']))
len(set(submission['fullVisitorId'])) == len(set(final_test['fullVisitorId']))
"""

"""
####
# explore what we are looking at
####

#need to go through and clean the columns
all_train.describe()

all_train.columns
#51 columns
all_train['adwordsClickInfo'][0] #this is still json buy okay
type(all_train['transactionRevenue'][0])  == np.float64#this is the one we are trying to predict
all_train.columns
"""

####
# scan columns and classify
####

numeric = []
categorical = []
flatline = []
other = []

for col in all_train.columns:
	if type(all_train[col][0]) == str:
		#categorical
		if len(all_train[col].unique()) > 1:
			categorical.append(col)
		else:
			flatline.append(col)
	elif type(all_train[col][0]) == int or type(all_train[col][0]) == np.float64:
		#numeric
		numeric.append(col)
	else:
		other.append(col)

numeric
categorical
flatline
other

####
# other columns
####
drop_other = ['visitId',
				'campaignCode',
				'referralPath',
				'adwordsClickInfo',
				'adContent']


numeric_other = ['visitNumber', 
					'hits',
					'visits']

categorical_other = ['isMobile',]



####
# drop flat cols for both the train and test data
####

flatline.extend(drop_other)
#should drop the flatline columns from the df
all_train = all_train.drop(flatline, axis = 1)
all_train.shape

flatline = [x for x in flatline if x != 'campaignCode' ]
final_test = final_test.drop(flatline, axis=1)
final_test.shape

for i in list(all_train.columns):
	if i not in list(final_test.columns):
		print(i)



####
# numeric
####
print('numeric variables')

#'fullVisitorId' #removed form numeric, this is just the id
#'transactionRevenue' #this is the response variable we want to predict

numeric = [ 'newVisits',
			 'pageviews',
			 'transactionRevenue',
			 ]

numeric.extend(numeric_other)

all_train['transactionRevenue'].fillna(0, inplace = True)

def fill_and_adj_numeric(df):
	#there are NA for page views, fill median for this == 1
	df.pageviews.fillna(df.pageviews.median(), inplace = True)

	df.hits.fillna(df.hits.median(), inplace = True)
	df.visits.fillna(df.visits.median(), inplace = True)

	#are boolean, fill NaN with zeros, add to categorical
	df.isTrueDirect.fillna(0, inplace = True)
	df.bounces.fillna(0, inplace = True)
	df.newVisits.fillna(0, inplace = True)
	df.visitNumber.fillna(1, inplace = True)

	for col in ['isTrueDirect', 'bounces', 'newVisits']:
		df[col] = df[col].astype(int)

	return df

all_train = fill_and_adj_numeric(all_train)
final_test = fill_and_adj_numeric(final_test)


####
# datetime columns - parse to separate features
##
print('Date variable')
all_train['date'] #this needs to be processed with datetime

def parseDateCol(df, date_col):
	""" takes the date column and adds new columns with the features:
		yr, mon, day, day of week, day of year """
	df['datetime'] = df.apply(lambda x : time.strptime(str(x[date_col]),  "%Y%M%d"), axis = 1)
	print('parsing year')
	df['year'] = df.apply(lambda x : x['datetime'].tm_year, axis = 1)
	print('parsing month')
	df['month'] = df.apply(lambda x :x['datetime'].tm_mon , axis = 1)
	print('parsing days (*3 versions)')
	df['mday'] = df.apply(lambda x : x['datetime'].tm_mday, axis = 1)
	df['wday'] = df.apply(lambda x : x['datetime'].tm_wday , axis = 1)
	df['yday'] = df.apply(lambda x : x['datetime'].tm_yday , axis = 1)

	#drop date and datetime
	df = df.drop([date_col, 'datetime'], axis = 1)
	
	return df

all_train = parseDateCol(all_train, 'date')

final_test = parseDateCol(final_test, 'date')



####
# categorical - one hot encode
####
print('Cleaning categorical variables')

categorical = ['channelGrouping',
				 'sessionId',
				 'browser',
				 'deviceCategory',
				 'operatingSystem',
				 'city',
				 'continent',
				 'country',
				 'metro',
				 'networkDomain',
				 'region',
				 'subContinent',
				 'campaign',
				 'keyword',
				 'medium',
				 'source']

categorical.extend(categorical_other)


with_na = []
for col in categorical:
	if all_train[col].isnull().any() :
		with_na.append(col)		

####
# fill na for all the categoricals with the 'None' if string or mode if bool
####

#most common value to fill the na
all_train.keyword.fillna('(not provided)', inplace = True)

def binarize_col(train, test, col):
	encoder = LabelBinarizer()

	cat_train_1hot = encoder.fit_transform(train[col])
	
	cat_test_1hot = encoder.transform(test[col])

	return cat_train_1hot, cat_test_1hot


train_bins = []
test_bins = []


for col in categorical:
	if len(all_train[col].unique()) > 1 and len(all_train[col].unique()) < 50:

		print(f'binarizing:{col}\tunique: {len(all_train[col].unique())}')

		bin_col_all_train, bin_col_final_test = binarize_col(all_train, final_test, col)

		if len(train_bins) == 0:
			print('initializing np matrix')
			train_bins = bin_col_all_train	
			test_bins =	bin_col_final_test
		else:
			print('appending to np matrix')
			train_bins = np.c_[train_bins, bin_col_all_train]
			test_bins = np.c_[test_bins, bin_col_final_test]
	gc.collect()


#drop the non binarized categorical columns and the housekeeping ones ones to the 
#the train and test sets for sklearn
all_train = all_train.drop(categorical, axis = 1)
final_test = final_test.drop(categorical, axis = 1)


# isolate the response variable
y_train = all_train['transactionRevenue'].values
#take the log1p on the front and end then use that to train the algorithm.
y_train =  np.log1p(y_train)

#drop the id and reponse variable from the matrix
X_train = all_train.drop(['fullVisitorId','transactionRevenue'], axis = 1).values
X_train = np.c_[X_train, train_bins]
X_train.shape


X_test = final_test.drop(['fullVisitorId'], axis = 1).values
X_test = np.c_[X_test, test_bins]
X_test.shape

#dump the np matrices to file
X_train.dump('X_train.dat')
y_train.dump('y_train.dat')
X_test.dump('X_test.dat')

