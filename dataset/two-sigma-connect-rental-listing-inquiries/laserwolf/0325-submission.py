import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = 8088
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20,
                         verbose_eval=25)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

def add_fields(train_df):
    # count of photos #
	train_df["num_photos"] = train_df["photos"].apply(len)

	# count of "features" #
	train_df["num_features"] = train_df["features"].apply(len)

	# count of words present in description column #
	train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))

	# convert the created column to datetime object so as to extract more features 
	train_df["created"] = pd.to_datetime(train_df["created"])

	# Let us extract some features like year, month, day, hour from date columns #
	train_df["created_year"] = train_df["created"].dt.year
	train_df["created_month"] = train_df["created"].dt.month
	train_df["created_day"] = train_df["created"].dt.day
	train_df["created_hour"] = train_df["created"].dt.hour

	train_df["created_day_hour"] = train_df["created"].dt.day*24 + train_df["created"].dt.hour

	train_df["created_dow"] = train_df["created"].dt.day%7
	train_df["created_dow_hour"] = train_df["created_dow"]*24 + train_df["created_hour"]

	train_df["price_per_bed"] = train_df["price"]/train_df["bedrooms"].clip(lower=1)
	train_df['created_date']=np.array(train_df.created.values, dtype='datetime64[D]'
					 ).astype(np.float32)

	train_df["created_date_hour"] = train_df["created_date"]*24+train_df["created"].dt.hour
	train_df['hour_count']=train_df.groupby("created_date_hour")['bathrooms'].transform(
	    lambda x: x.count())
	train_df['date_count']=train_df.groupby("created_date_hour")['bathrooms'].transform(
	    lambda x: x.count())

def add_manager(train_df_tr, train_df_te):
    temp = pd.concat([train_df_tr.manager_id,pd.get_dummies(train_df_tr.interest_level)], axis = 1
                    ).groupby('manager_id').mean()
    temp.columns = ['high_frac','low_frac', 'medium_frac']
    temp['count'] = train_df_tr.groupby('manager_id').count().iloc[:,1]

    temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
    unranked_managers_ixes = temp['count']<20
    ranked_managers_ixes = ~unranked_managers_ixes
    mean_values = temp.loc[ranked_managers_ixes, [
        'high_frac','low_frac', 'medium_frac','manager_skill']].mean()
    temp.loc[unranked_managers_ixes,[
        'high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values

    train_df_tr = train_df_tr.merge(temp.reset_index(),how='left', on='manager_id')
    train_df_te = train_df_te.merge(temp.reset_index(),how='left', on='manager_id')
    new_manager_ixes = train_df_te['high_frac'].isnull()
    train_df_te.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill'
                                ]] = mean_values.values
    return (train_df_tr, train_df_te)
    
data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

add_fields(train_df)
add_fields(test_df)

hour_counts_df=pd.concat([train_df, test_df]).groupby('created_date_hour')['listing_id'].agg([
    'min', 'max', 'count'])
hour_counts_df['min_next']=hour_counts_df['min'].shift(-1)

hour_counts_df.loc[(hour_counts_df['min']>hour_counts_df[
    'min_next']), 'min_next']=hour_counts_df.loc[(
    hour_counts_df['min']>hour_counts_df['min_next']), 'max']+1
hour_counts_df.loc[(hour_counts_df['min']+10000<hour_counts_df[
    'min_next']), 'min_next']=hour_counts_df.loc[(
    hour_counts_df['min']+10000<hour_counts_df['min_next']), 'max']+1

hour_counts_df.loc[hour_counts_df['min_next'].isnull(), 'min_next'] = hour_counts_df.loc[
    hour_counts_df['min_next'].isnull(), 'min']+225
hour_counts_df['full_hour_count']=hour_counts_df['min_next']-hour_counts_df['min']

hour_counts_df['count_ratio']=hour_counts_df.apply(lambda x: x['count']/x['full_hour_count'], axis=1)

train_df=train_df.merge(hour_counts_df[['full_hour_count']], left_on=['created_date_hour'], 
                        right_index=True)
test_df=test_df.merge(hour_counts_df[['full_hour_count']], left_on=['created_date_hour'], 
                       right_index=True)

categorical = ["display_address", "manager_id", "building_id", "street_address"]

for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
print(train_df["features"].head())
tfidf = CountVectorizer(stop_words='english', max_features=200)
tfidf.fit(train_df["features"])

'''import re
cleanr = re.compile('<.*?>')
train_df['description']=train_df['description'].apply(lambda x: re.sub(cleanr, ' ', x))
tfidf2 = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english', ngram_range=(3, 5), max_features=1000)
tfidf2.fit(train_df.description.values)'''

train_df_tr, train_df_te = train_df, test_df
tr_sparse_tr = tfidf.transform(train_df_tr["features"])
tr_sparse_te = tfidf.transform(train_df_te["features"])

features_sparse = ['features_'+str(i) for i in range(tr_sparse_tr.shape[1])]

#te_sparse = tfidf.transform(test_df["features"])
(train_df_tr, train_df_te) = add_manager(train_df_tr, train_df_te)

features_to_use=['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'num_photos', 
                 'num_features', 'num_description_words', #'created_year', 'created_month', 
                 #'created_day', 'listing_id', 'created_hour', 'price_per_bed', 'created_date', 
                 'listing_id','price_per_bed',
                 'display_address', 'manager_id', 'building_id',
                 'manager_skill',
                  'created_dow', 'created_hour', 'full_hour_count']
#features_to_use+=["created_dow"+str(i) for i in range(7)]
print(features_to_use)

train_X_tr = sparse.hstack([train_df_tr[features_to_use], tr_sparse_tr]).tocsr()
#, tr_sparse2_tr]).tocsr()
train_X_te = sparse.hstack([train_df_te[features_to_use], tr_sparse_te]).tocsr()
#, tr_sparse2_te]).tocsr()
#test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y_tr = np.array(train_df_tr['interest_level'].apply(lambda x: target_num_map[x]))

print(train_X_tr.shape, train_X_te.shape)#, test_X.shape)

train_X_tr_n=train_X_tr.toarray()
train_X_te_n=train_X_te.toarray()

print('training')
preds, model = runXGB(train_X_tr_n, train_y_tr, train_X_te_n, num_rounds=225)
print('writing output file')
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = train_df_te.listing_id.values
out_df.to_csv("0325.1.csv", index=False)

print('done')

