import os
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
import copy



class Munger():
	def __init__(self, basedir='~/Kaggle/airbnb/data'):
		np.random.seed(0)
		basedir = os.path.expanduser(basedir)
		self.le = LabelEncoder()
		
		#Loading data
		df_train = pd.read_csv(os.path.join(basedir, 'train_users_2.csv'))
		df_test = pd.read_csv(os.path.join(basedir, 'test_users.csv'))
		self.labels = df_train['country_destination'].values
		df_train = df_train.drop(['country_destination'], axis=1)
		self.id_test = df_test['id']
		self.piv_train = df_train.shape[0]

		#Creating a DataFrame with train+test data
		df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
		#Removing id and date_first_booking
		df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
		#Filling nan
		self.df_all = df_all.fillna(-1)
		
	def engineer_features(self, df_all=None):
		if df_all==None:
			df_all = self.df_all
		#date_account_created
		dac = pd.to_datetime(df_all['date_account_created'])
		df_all['dac_year']  = dac.apply( lambda x: x.year)
		df_all['dac_month'] = dac.apply( lambda x: x.month)
		df_all['dac_day']   = dac.apply( lambda x: x.day)


		def convert_time(time_string):
			return datetime.datetime.strptime(str(time_string), '%Y%m%d%H%M%S')

		#timestamp_first_active
		tfa = df_all.timestamp_first_active.apply(convert_time)
		df_all['tfa_year']  = tfa.apply( lambda x: x.year)
		df_all['tfa_month'] = tfa.apply( lambda x: x.month)
		df_all['tfa_day']   = tfa.apply( lambda x: x.day)
		df_all['tfa_hour']  = tfa.apply( lambda x: x.hour)

		#timedelta
		df_all['timedelta'] =  pd.to_datetime(tfa - dac).apply(lambda x: x.day)

		df_all = df_all.drop(['timestamp_first_active'], axis=1)
		df_all = df_all.drop(['date_account_created'], axis=1)
		
		def myround(x, base=5):
		    if x < 14 or x > 70:
			    return -1
		    return int(base * round(float(x)/base))
		
		#Age
		df_all['age'] =  df_all.age.fillna(-1).apply(myround)

		self.df_all = df_all

	def one_hot_encode(self, df_all=None): 
		if df_all==None:
			df_all = self.df_all
		#One-hot-encoding features
		ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
		for f in ohe_feats:
		    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
		    df_all = df_all.drop([f], axis=1)
		    df_all = pd.concat((df_all, df_all_dummy), axis=1)

		self.df_all = df_all

	def label_transformer(self, labels=None):
		if labels == None:
			labels = self.labels
			return self.le.fit_transform(labels) 
		return self.le.transform(labels) 

	def label_inverse_transformer(self, labels):
		return self.le.inverse_transform(labels) 

	def data_split(self, df_all=None):
		if df_all == None:
			df_all = self.df_all
		vals = df_all.values
		X = vals[:self.piv_train]
		X_test = vals[self.piv_train:]

		return X, X_test,

	def get_train_data(self):
		df_all = copy.deepcopy(self.df_all.iloc[:self.piv_train])
		df_all['y'] = self.labels
		return df_all

	

def clean_data():
	M = Munger()
	M.engineer_features()
	#M.one_hot_encode() 
	return M