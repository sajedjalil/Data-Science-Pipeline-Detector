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
from sklearn.model_selection import train_test_split

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
    
data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)
train_df["photos_per_bed"] = train_df["photos"].apply(len)/train_df["bedrooms"].clip(lower=1)
test_df["photos_per_bed"] = test_df["photos"].apply(len)/test_df["bedrooms"].clip(lower=1)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])

# Let us extract some features like year, month, day, hour from date columns #
train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour

train_df["price_per_bed"] = train_df["price"]/train_df["bedrooms"].clip(lower=1)
test_df["price_per_bed"] = test_df["price"]/test_df["bedrooms"].clip(lower=1)
train_df['created_date']=np.array(train_df.created.values, dtype='datetime64[D]'
                                 ).astype(np.float32)
test_df['created_date']=np.array(test_df.created.values, dtype='datetime64[D]'
                                 ).astype(np.float32)
train_df['created_dow']=np.array(train_df.created.values, dtype='datetime64[D]'
                                 ).astype(np.float32)%7
test_df['created_dow']=np.array(test_df.created.values, dtype='datetime64[D]'
                                 ).astype(np.float32)%7

# adding all these new features to use list #
features_to_use.extend(["num_photos", "num_features", "num_description_words",
                        "created_month", "listing_id", "created_hour", "photos_per_bed", 
                        "price_per_bed", "created_date"])
                        
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)
            
train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
print(train_df["features"].head())
tfidf = CountVectorizer(stop_words='english', max_features=200)
tfidf.fit(train_df["features"])

train_df_tr=train_df
tr_sparse_tr = tfidf.transform(train_df_tr["features"])
te_sparse = tfidf.transform(test_df["features"])

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

test_df = test_df.merge(temp.reset_index(),how='left', on='manager_id')
new_manager_ixes = test_df['high_frac'].isnull()
test_df.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill'
                            ]] = mean_values.values

features_to_use.extend(['manager_skill'])

print(features_to_use)

train_X_tr = sparse.hstack([train_df_tr[features_to_use], tr_sparse_tr]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

features_sparse = ['features_'+str(i) for i in range(tr_sparse_tr.shape[1])]
target_num_map = {'high':0, 'medium':1, 'low':2}
train_y_tr = np.array(train_df_tr['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X_tr.shape, test_X.shape)

preds, model = runXGB(train_X_tr, train_y_tr, test_X, num_rounds=325)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("0331.2.csv", index=False)
