import pandas as pd
import csv
import math
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import EarlyStopping
import lightgbm as lgb
import gc, sys
gc.enable()

dtypes = {
		'assists'           : 'uint8',
		'boosts'            : 'uint8',
		'damageDealt'       : 'float16',
		'DBNOs'             : 'uint8',
		'headshotKills'     : 'uint8', 
		'heals'             : 'uint8',    
		'killPlace'         : 'uint8',    
		'killPoints'        : 'uint8',    
		'kills'             : 'uint8',    
		'killStreaks'       : 'uint8',    
		'longestKill'       : 'float16',    
		'maxPlace'          : 'uint8',    
		'numGroups'         : 'uint8',    
		'revives'           : 'uint8',    
		'rideDistance'      : 'float16',    
		'roadKills'         : 'uint8',    
		'swimDistance'      : 'float16',    
		'teamKills'         : 'uint8',    
		'vehicleDestroys'   : 'uint8',    
		'walkDistance'      : 'float16',    
		'weaponsAcquired'   : 'uint8',    
		'winPoints'         : 'uint8', 
		'winPlacePerc'      : 'float16' 
}

def feature_engineering(filename,train = False):
	data = pd.read_csv(filename,dtype=dtypes)
	data = data[data['maxPlace'] > 1]
	data['headshotrate'] = data['kills']/data['headshotKills']
	data['killStreakrate'] = data['killStreaks']/data['kills']
	data['healthitems'] = data['heals'] + data['boosts']
	data['totalDistance'] = data['rideDistance'] + data["walkDistance"] + data["swimDistance"]
	data['killPlace_over_maxPlace'] = data['killPlace'] / data['maxPlace']
	data['headshotKills_over_kills'] = data['headshotKills'] / data['kills']
	data['distance_over_weapons'] = data['totalDistance'] / data['weaponsAcquired']
	data['walkDistance_over_heals'] = data['walkDistance'] / data['heals']
	data['walkDistance_over_kills'] = data['walkDistance'] / data['kills']
	data['killsPerWalkDistance'] = data['kills'] / data['walkDistance']
	data["skill"] = data["headshotKills"] + data["roadKills"] 
	data[data == np.Inf] = np.NaN
	data[data == np.NINF] = np.NaN

	data.fillna(0, inplace=True)
	feature = list(data.columns)
	feature.remove('Id')
	feature.remove('groupId')
	feature.remove('matchId')
	feature.remove('matchType')
	if(train):
		labels = np.array(data.groupby(['matchId','groupId'])['winPlacePerc'].agg('mean'), dtype=np.float64)
		feature.remove('winPlacePerc')
	else: 
		labels = data[['Id']]
	
	print("group_max")
	agg = data.groupby(['matchId','groupId'])[feature].agg('max')
	agg_rank = agg.groupby('matchId')[feature].rank(pct=True).reset_index()
	if train: data_out = agg.reset_index()[['matchId','groupId']]
	else: data_out = data[['matchId','groupId']]
	data_out = data_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId','groupId'])
	data_out = data_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId','groupId'])
	
	print("group_mean")
	agg = data.groupby(['matchId','groupId'])[feature].agg('mean')
	agg_rank = agg.groupby('matchId')[feature].rank(pct=True).reset_index()
	data_out = data_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId','groupId'])
	data_out = data_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId','groupId'])
	
	print("group_min")
	agg = data.groupby(['matchId','groupId'])[feature].agg('min')
	agg_rank = agg.groupby('matchId')[feature].rank(pct=True).reset_index()
	data_out = data_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId','groupId'])
	data_out = data_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId','groupId'])
	
	print("match_mean")
	agg = data.groupby(['matchId'])[feature].agg('mean').reset_index()
	data_out = data_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
	
	print("match_max")
	agg = data.groupby(['matchId'])[feature].agg('max').reset_index()
	data_out = data_out.merge(agg, suffixes=["", "_match_max"], how='left', on=['matchId'])
	
	print("match_size")
	agg = data.groupby(['matchId']).size().reset_index(name='match_size')
	data_out = data_out.merge(agg, how='left', on=['matchId'])
	

	del data,agg,agg_rank
	gc.collect()
	data_out.drop(["matchId", "groupId"], axis=1, inplace=True)

	data_out = reduce_size(data_out)
	X = data_out
	del data_out, feature
	gc.collect()
	return X,labels

def reduce_size(merged_data_out):
	print('      Starting size is %d Mb'%(sys.getsizeof(merged_data_out)/1024/1024))
	print('      Columns: %d'%(merged_data_out.shape[1]))
	feats = merged_data_out.columns[merged_data_out.dtypes == 'float64']
	for feat in feats:
		merged_data_out[feat] = merged_data_out[feat].astype('float32')

	feats = merged_data_out.columns[merged_data_out.dtypes == 'int16']
	for feat in feats:
		mm = np.abs(merged_data_out[feat]).max()
		if mm < 126:
			merged_data_out[feat] = merged_data_out[feat].astype('int8')

	feats = merged_data_out.columns[merged_data_out.dtypes == 'int32']
	for feat in feats:
		mm = np.abs(merged_data_out[feat]).max()
		if mm < 126:
			merged_data_out[feat] = merged_data_out[feat].astype('int8')
		elif mm < 30000:
			merged_data_out[feat] = merged_data_out[feat].astype('int16')

	feats = merged_data_out.columns[merged_data_out.dtypes == 'int64']
	for feat in feats:
		mm = np.abs(merged_data_out[feat]).max()
		if mm < 126:
			merged_data_out[feat] = merged_data_out[feat].astype('int8')
		elif mm < 30000:
			merged_data_out[feat] = merged_data_out[feat].astype('int16')
		elif mm < 2000000000:
			merged_data_out[feat] = merged_data_out[feat].astype('int32')
	print('      Ending size is %d Mb'%(sys.getsizeof(merged_data_out)/1024/1024))
	return merged_data_out



params = {
	'objective': 'regression',
	'early_stopping_rounds':200,
	'n_estimators':20000,
	'metric': 'mae',
	"bagging_seed" : 0,
	'num_leaves': 31,
	'learning_rate': 0.05,
	'bagging_fraction': 0.9,
	"num_threads" : 4,
	"colsample_bytree" : 0.7
}

if __name__ == '__main__':
	batch_size = 512
	num_of_features  = 0
	#features = load_csv_data('../input/train_V2.csv')
	#test = load_csv_data('../input/test_V2.csv')
	trainpath = '../input/train_V2.csv'
	testpath = '../input/test_V2.csv'
	features, labels = feature_engineering(trainpath,train = True)
	num_of_features = features.shape[1]
	
	filepath = "best.h5"
	split = int(len(labels)*0.8)
	lgb_train = lgb.Dataset(features[:split], labels[:split])
	lgb_val = lgb.Dataset(features[split:], labels[split:])
	del features, labels
	gc.collect()
	gbm = lgb.train(params, lgb_train, verbose_eval=100,valid_sets=[lgb_train, lgb_val],early_stopping_rounds = 200)
	del lgb_train,lgb_val
	gc.collect()

	features_test, test = feature_engineering(testpath)
	predict = gbm.predict(features_test,num_iteration=gbm.best_iteration)
	del features_test
	gc.collect()
	predict = predict.reshape(-1)
	test['winPlacePerc'] = predict


	df_test = pd.read_csv(testpath)

	# Restore some columns
	test = test.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

	# Sort, rank, and assign adjusted ratio
	df_sub_group = test.groupby(["matchId", "groupId"]).first().reset_index()
	df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
	df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)


	test = test.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
	test["winPlacePerc"] = test["adjusted_perc"]

	# Deal with edge cases
	test.loc[test.maxPlace == 0, "winPlacePerc"] = 0
	test.loc[test.maxPlace == 1, "winPlacePerc"] = 1

	# Align with maxPlace
	# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
	subset = test.loc[test.maxPlace > 1]
	gap = 1.0 / (subset.maxPlace.values - 1)
	new_perc = np.around(subset.winPlacePerc.values / gap) * gap
	test.loc[test.maxPlace > 1, "winPlacePerc"] = new_perc

	# Edge case
	test.loc[(test.maxPlace > 1) & (test.numGroups == 1), "winPlacePerc"] = 0
	assert test["winPlacePerc"].isnull().sum() == 0

	test[["Id", "winPlacePerc"]].to_csv("submission_adjusted.csv", index=False)