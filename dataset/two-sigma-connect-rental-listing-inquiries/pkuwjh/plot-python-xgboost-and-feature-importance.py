import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig

import random
import operator
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

print("Started")

RS = 2016
random.seed(RS)
np.random.seed(RS)

input_folder = '../input/'
interest_level = ['high', 'medium', 'low']
n_class = len(interest_level)
interest_level_dict = {w:i for i, w in enumerate(interest_level)}

def train_xgb(X, y, params):
	#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=RS)
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.010, random_state=RS)

	xg_train = xgb.DMatrix(X_train, label=y_train)
	xg_val = xgb.DMatrix(X_val, label=y_val)

	watchlist  = [(xg_train,'train'), (xg_val,'eval')]
	#return xgb.train(params, xg_train, params['num_rounds'], watchlist)
	return xgb.train(params, xg_train, params['num_rounds'])

def predict_xgb(clr, X_test):
	return clr.predict(xgb.DMatrix(X_test))

def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	i = 0
	for feat in features:
		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
		i = i + 1

	outfile.close()

def main():
	params = {}
	params['objective'] = 'multi:softprob'
	params['eval_metric'] = 'mlogloss'
	params['num_class'] = n_class
	params['eta'] = 0.08
	params['max_depth'] = 6
	params['subsample'] = 0.7
	params['colsample_bytree'] = 0.7
	params['silent'] = 1
	params['num_rounds'] = 350
	params['seed'] = RS

	d_train = pd.read_json(input_folder + 'train.json')
	d_test = pd.read_json(input_folder + 'test.json')
	
	y_train = d_train['interest_level'].replace(interest_level_dict)
	X_train = d_train
	del X_train['interest_level']
	X_test = d_test
	ids = d_test['listing_id'].values
	print("Original data: X_train: {}, y_train: {}, X_test: {}".format(X_train.shape, y_train.shape, X_test.shape))
	
	train_len = X_train.shape[0]
	df = pd.concat([X_train, X_test])
	
	df['num_photos'] = df['photos'].apply(len)
	df['num_features'] = df['features'].apply(len)
	df['num_description_words'] = df['description'].apply(lambda x: len(x.split(' ')))
	df['num_description_len'] = df['description'].apply(len)

	desc_feats = {'bathroom_mentions': ['bathroom', 'bthrm', 'ba '],
				  'bedroom_mentions': ['bedroom', 'bdrm', 'br '],
				  'kitchen_mentions': ['kitchen', 'kit ']}

	for name, kwords in desc_feats.items():
		df[name] =  df['description'].apply(lambda x: sum([x.count(w) for w in kwords]))

	df['created'] = pd.to_datetime(df['created'])
	df['created_month'] = df['created'].dt.month
	df['created_day'] = df['created'].dt.day
	df['created_hour'] = df['created'].dt.hour
	df['created_weekday'] = df['created'].dt.weekday
	df['created_week'] = df['created'].dt.week
	df['created_quarter'] = df['created'].dt.quarter
	#df['created_wd'] = ((df['created_weekday'] != 5) & (df['created_weekday'] != 6))
	print(df.head(1))

	categorical = ["display_address", "manager_id", "building_id", "street_address"]
	for f in categorical:
		df[f] = LabelEncoder().fit_transform(df[f])
	
	not_feats = ['created','description','features','listing_id','photos']
	is_feats = [col for col in df.columns if col not in not_feats]
	print("Features: {}".format(is_feats))
	
	df['features'] = df["features"].apply(lambda x: " ".join(x))
	tfidf = TfidfVectorizer(stop_words='english', max_features=150)
	df_sparse = tfidf.fit_transform(df["features"])

	df = sparse.hstack([df[is_feats], df_sparse]).tocsr()
	
	X_train = df[:train_len]
	X_test = df[train_len:]
	del df
	
	feature_names = is_feats + ['sparse_%d' % i for i in range(df_sparse.shape[1])]
	create_feature_map(feature_names)
	
	print("Training on: X_train: {}, y_train: {}, X_test: {}".format(X_train.shape, y_train.shape, X_test.shape))
	clr = train_xgb(X_train, np.array(y_train.astype(np.int8)), params)
	preds = predict_xgb(clr, X_test)

	print("Writing submission file...")
	with open('my_xgb.csv', 'w') as wf:
		wf.write('listing_id,{}\n'.format(','.join(interest_level)))
		for i, pred in enumerate(preds):
			wf.write('{},{}\n'.format(ids[i], ','.join(map(str, pred))))

	print("Submission file done, plotting features importance...")
	importance = clr.get_fscore(fmap='xgb.fmap')
	importance = sorted(importance.items(), key=operator.itemgetter(1))

	df = pd.DataFrame(importance, columns=['feature', 'fscore'])

	plt.figure()
	df.plot()
	df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
	plt.gcf().savefig('features_importance.png')

main()
print("the process have been Done!")