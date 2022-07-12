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
from sklearn.feature_extraction.text import TfidfVectorizer
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=333):
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
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=25)
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


# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)



# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

train_df['photos_features']=train_df["num_features"]*train_df["num_description_words"]
test_df['photos_features']=test_df["num_features"]*test_df["num_description_words"]
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
train_df["Hardwood Floors"]=train_df["features"].apply(lambda x: int("Hardwood Floors" in x))
train_df["No Fee"]=train_df["features"].apply(lambda x: int("No Fee" in x))
train_df["Dishwasher"]=train_df["features"].apply(lambda x: int("Dishwasher" in x))
train_df['Dogs Allowed']=train_df["features"].apply(lambda x: int('Dogs Allowed' in x))
train_df['Doorman']=train_df["features"].apply(lambda x: int('Doorman' in x))
train_df['Fitness Center']=train_df["features"].apply(lambda x: int('Fitness Center' in x))
train_df['Cats Allowed']=train_df["features"].apply(lambda x: int('Cats Allowed' in x))
train_df['Elevator']=train_df["features"].apply(lambda x: int('Elevator' in x))
train_df['building']=train_df['description'].apply(lambda x: int('building' in x.split(' ')))
train_df['kitchen']=train_df['description'].apply(lambda x:int('kitchen' in x.split(' ')))
train_df['low_features']=train_df['Dogs Allowed']+train_df['Cats Allowed']+train_df['Doorman']
train_df['medium_features']=train_df['Dishwasher']+train_df['Elevator']
train_df['high_features']=train_df['No Fee']+train_df['Elevator']
train_df['all_features']=train_df['low_features']+train_df['medium_features']+train_df['high_features']
train_df['num_room']=train_df['bathrooms']+train_df['bedrooms']
train_df['rate_room']=train_df['num_room']*1.0/train_df['price']
train_df['rate_photos']=train_df['num_photos']*1.0/train_df['price']
train_df['rate_words']=train_df['num_description_words']*1.0/train_df['price']

test_df["Hardwood Floors"]=test_df["features"].apply(lambda x: int("Hardwood Floors" in x))
test_df["No Fee"]=test_df["features"].apply(lambda x: int("No Fee" in x))
test_df["Dishwasher"]=test_df["features"].apply(lambda x: int("Dishwasher" in x))
test_df['Dogs Allowed']=test_df["features"].apply(lambda x: int('Dogs Allowed' in x))
test_df['Doorman']=test_df["features"].apply(lambda x: int('Doorman' in x))
test_df['Fitness Center']=test_df["features"].apply(lambda x: int('Fitness Center' in x))
test_df['Cats Allowed']=test_df["features"].apply(lambda x: int('Cats Allowed' in x))
test_df['Elevator']=test_df["features"].apply(lambda x: int('Elevator' in x))
test_df['building']=test_df['description'].apply(lambda x: int('building' in x.split(' ')))
test_df['kitchen']=test_df['description'].apply(lambda x:int('kitchen' in x.split(' ')))
test_df['low_features']=test_df['Dogs Allowed']+test_df['Cats Allowed']+test_df['Doorman']
test_df['medium_features']=test_df['Dishwasher']+test_df['Elevator']
test_df['high_features']=test_df['No Fee']+test_df['Elevator']
test_df['num_room']=test_df['bathrooms']+test_df['bedrooms']
test_df['all_features']=test_df['low_features']+test_df['medium_features']+test_df['high_features']
test_df['rate_room']=test_df['num_room']*1.0/test_df['price']
test_df['rate_photos']=test_df['num_photos']*1.0/test_df['price']
test_df['rate_words']=test_df['num_description_words']*1.0/test_df['price']



features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]
# adding all these new features to use list #
features_to_use.extend(["num_photos", "num_features", "num_description_words","created_year", "created_month", "created_day", 
	 "listing_id","created_hour",'rate_room'])

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
tfidf = CountVectorizer(stop_words='english', max_features=205)
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

preds, model = runXGB(train_X, train_y, test_X, num_rounds=333)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("xgb_starter2.csv", index=False)