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

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=10):
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
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model
  
def displayAddressLocation(df):
    df['west_address'] = df['display_address'].str.contains('west').astype(int)
    df['east_address'] = df['display_address'].str.contains('east').astype(int)
    df['south_address'] = df['display_address'].str.contains('south').astype(int)
    df['north_address'] = df['display_address'].str.contains('north').astype(int)
    df['avenue_address'] = df['display_address'].str.contains('ave').astype(int)
    df['street_address'] = df['display_address'].str.contains('st').astype(int)
    df['place_address'] = df['display_address'].str.contains('place').astype(int)
    df['broadway_address'] = df['display_address'].str.contains('broadway').astype(int)
    df['parkway_address'] = df['display_address'].str.contains('parkway|pkwy').astype(int)
    
    return df

def adjustDisplayAddress(df):
    # change display address to lower case #
    df['display_address'] = df['display_address'].str.lower()
    
    # remove . in display address #
    df['display_address'] = df['display_address'].str.replace('.', '')
    
    # remove second in display address #
    df['display_address'] = df['display_address'].str.replace('second', '2')
    
    # remove third in display address #
    df['display_address'] = df['display_address'].str.replace('third', '3')
    
    # remove forth in display address #
    df['display_address'] = df['display_address'].str.replace('forth', '4')
    
    # remove fifth in display address #
    df['display_address'] = df['display_address'].str.replace('fifth', '5')
    
    # remove sixth in display address #
    df['display_address'] = df['display_address'].str.replace('sixth', '6')
    
    # remove seventh in display address #
    df['display_address'] = df['display_address'].str.replace('seventh', '7')
    
    # remove eighth in display address #
    df['display_address'] = df['display_address'].str.replace('eighth', '8')
    
    # remove ninth in display address #
    df['display_address'] = df['display_address'].str.replace('ninth', '9')
    
    # remove tenth in display address #
    df['display_address'] = df['display_address'].str.replace('tenth', '10')

    # change 3rd to 3 #
    df['display_address'] = df['display_address'].str.replace('3rd', '3')
    
    # change 2nd to 2 #
    df['display_address'] = df['display_address'].str.replace('2nd', '2')
    
    # change first to 1st #
    df['display_address'] = df['display_address'].str.replace('first', ' 1st')
    
    # change 1st to 1 #
    df['display_address'] = df['display_address'].str.replace('1st', ' 1')
    
    df = displayAddressLocation(df)
    
    # remove th in display address #
    df['display_address'] = df['display_address'].str.replace('th', '')
    
    # change "avenue","ave.","ave" to a #
    df['display_address'] = df['display_address'].str.replace('avenue', 'a')
    df['display_address'] = df['display_address'].str.replace('ave', 'a')
    
    # change "street","st." to st #
    df['display_address'] = df['display_address'].str.replace('street', 'st')
    
    # change "parkway" to pkwy #
    df['display_address'] = df['display_address'].str.replace('parkway', 'pkwy')
    
    # change "east" to e #
    df['display_address'] = df['display_address'].str.replace('east', 'e')
    
    # change "west" to w #
    df['display_address'] = df['display_address'].str.replace('west', 'w')
    
    # change "north" to n #
    df['display_address'] = df['display_address'].str.replace('north', 'n')
    
    # change "south" to s #
    df['display_address'] = df['display_address'].str.replace('south', 's')
    
    return df  
    
data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

#Play with price
train_df["price_bed"] = train_df["price"]/train_df["bedrooms"]
train_df["price_bath"] = train_df["price"]/train_df["bathrooms"]
train_df["price_bath_bed"] = train_df["price"]/(train_df["bathrooms"] + train_df["bedrooms"])

train_df["bedPerbath"] = (train_df["bedrooms"]/train_df["bathrooms"])
train_df["bedDiffbath"] = (train_df["bedrooms"]-train_df["bathrooms"])
train_df["bedPlusbath"] = (train_df["bedrooms"]+train_df["bathrooms"])
train_df["bedPerc"] = (train_df["bedrooms"]/(train_df["bathrooms"]+train_df["bedrooms"]))

train_df.fillna(-1,inplace=True)

test_df["price_bed"] = test_df["price"]/test_df["bedrooms"]
test_df["price_bath"] = test_df["price"]/test_df["bathrooms"]
test_df["price_bath_bed"] = test_df["price"]/(test_df["bathrooms"] + test_df["bedrooms"])

test_df["bedPerbath"] = (test_df["bedrooms"]/test_df["bathrooms"])
test_df["bedDiffbath"] = (test_df["bedrooms"]-test_df["bathrooms"])
test_df["bedPlusbath"] = (test_df["bedrooms"]+test_df["bathrooms"])
test_df["bedPerc"]     = (test_df["bedrooms"]/(test_df["bathrooms"]+test_df["bedrooms"]))

test_df.fillna(-1,inplace=True)

#Outliers
train_df.loc[train_df['price'] == 111111, 'price'] = 1025

features_to_use.extend(["price_bed", "price_bath", "price_bath_bed","bedPerbath", 
    "bedDiffbath", "bedPlusbath", "bedPerc", "price_bed","price_bath",
    "price_bath_bed", "bedPerbath", "bedDiffbath", "bedPlusbath","bedPerc"])

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

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

train_df["created_dayofyear"] = pd.to_datetime(train_df["created"]).dt.dayofyear
test_df["created_dayofyear"] = pd.to_datetime(test_df["created"]).dt.dayofyear

train_df["zero_building_id"] = train_df["building_id"].apply(lambda x: 1 if x=='0' else 0)
test_df["zero_building_id"] = test_df["building_id"].apply(lambda x: 1 if x=='0' else 0)

# adding all these new features to use list #
features_to_use.extend(["num_photos", "num_features", "num_description_words","created_year", "created_month", "created_day", "listing_id", "created_hour","created_dayofyear"])

# reduce display address #
train_df = adjustDisplayAddress(train_df)
test_df = adjustDisplayAddress(test_df)
    
# adding all these new features to use list #
features_to_use.extend(["west_address", "east_address","south_address", "north_address",
    "avenue_address", "street_address","place_address", "broadway_address","parkway_address"])

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
tfidf = CountVectorizer(stop_words='english', max_features=2)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)

cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break
    
preds, model = runXGB(train_X, train_y, test_X, num_rounds=4)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
print (out_df)
out_df.to_csv("xgb_starter2.csv", index=False)
print ('Finish...')